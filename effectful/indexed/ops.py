import functools
import numbers
import operator
import typing
from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import pyro
import torch

import effectful.internals.sugar

from ..internals.sugar import gensym, partial_eval, sizesof, torch_getitem
from ..ops.core import Expr, Operation, Term
from ..ops.function import defun

K = TypeVar("K")
T = TypeVar("T")


class IndexSet(Dict[str, Set[int]]):
    """
    :class:`IndexSet` s represent the support of an indexed value, primarily
    those created using :func:`intervene` and :class:`MultiWorldCounterfactual`
    for which free variables correspond to single interventions and indices
    to worlds where that intervention either did or did not happen.

    :class:`IndexSet` can be understood conceptually as generalizing :class:`torch.Size`
    from multidimensional arrays to arbitrary values, from positional to named dimensions,
    and from bounded integer interval supports to finite sets of positive integers.

    :class:`IndexSet`s are implemented as :class:`dict`s with
    :class:`Operation`s as keys corresponding to names of free index variables
    and :class:`set` s of positive :class:`int` s as values corresponding
    to the values of the index variables where the indexed value is defined.

    For example, the following :class:`IndexSet` represents
    the sets of indices of the free variables ``x`` and ``y``
    for which a value is defined::

        >>> IndexSet(x={0, 1}, y={2, 3})
        IndexSet({'x': {0, 1}, 'y': {2, 3}})

    :class:`IndexSet` 's constructor will automatically drop empty entries
    and attempt to convert input values to :class:`set` s::

        >>> IndexSet(x=[0, 0, 1], y=set(), z=2)
        IndexSet({'x': {0, 1}, 'z': {2}})

    :class:`IndexSet` s are also hashable and can be used as keys in :class:`dict` s::

        >>> indexset = IndexSet(x={0, 1}, y={2, 3})
        >>> indexset in {indexset: 1}
        True
    """

    def __init__(self, **mapping: Union[int, Iterable[int]]):
        index_set = {}
        for k, vs in mapping.items():
            indexes = {vs} if isinstance(vs, int) else set(vs)
            if len(indexes) > 0:
                index_set[k] = indexes
        super().__init__(**index_set)

    def __repr__(self):
        return f"{type(self).__name__}({super().__repr__()})"

    def __hash__(self):
        return hash(frozenset((k, frozenset(vs)) for k, vs in self.items()))

    def _to_handler(self):
        """Return an effectful handler that binds each index variable to a tensor of its possible index values."""
        return {
            k: functools.partial(lambda v: v, torch.tensor(list(v)))
            for k, v in self.items()
        }


def union(*indexsets: IndexSet) -> IndexSet:
    """
    Compute the union of multiple :class:`IndexSet` s
    as the union of their keys and of value sets at shared keys.

    If :class:`IndexSet` may be viewed as a generalization of :class:`torch.Size`,
    then :func:`union` is a generalization of :func:`torch.broadcast_shapes`
    for the more abstract :class:`IndexSet` data structure.

    Example::

        >>> s = union(IndexSet(a={0, 1}, b={1}), IndexSet(a={1, 2}))
        >>> {k:s[k] for k in sorted(s.keys())}
        {'a': {0, 1, 2}, 'b': {1}}

    .. note::

        :func:`union` satisfies several algebraic equations for arbitrary inputs.
        In particular, it is associative, commutative, idempotent and absorbing::

            union(a, union(b, c)) == union(union(a, b), c)
            union(a, b) == union(b, a)
            union(a, a) == a
            union(a, union(a, b)) == union(a, b)
    """
    return IndexSet(
        **{
            k: set.union(*[vs[k] for vs in indexsets if k in vs])
            for k in set.union(*(set(vs) for vs in indexsets))
        }
    )


@functools.lru_cache(maxsize=None)
def name_to_sym(name: str) -> Operation[[], int]:
    return gensym(int, name=name)


def lift_tensor(tensor: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, List[Operation]]:
    """Lift a tensor to an indexed tensor using the plate structure.

    Parameters:
    - tensor (torch.Tensor): A tensor.
    - name_to_dim: A dictionary mapping names to dimensions. If not provided, the plates returned by get_index_plates()
    are used.

    """
    name_to_dim = cast(
        dict[str, int],
        (
            kwargs["name_to_dim"]
            if "name_to_dim" in kwargs and kwargs["name_to_dim"] is not None
            else {name: f.dim for name, f in get_index_plates().items()}
        ),
    )
    event_dim = kwargs.get("event_dim", 0)

    index_expr: List[Any] = [slice(None)] * len(tensor.shape)
    for name, dim in name_to_dim.items():
        # ensure that lifted tensors use the same free variables for the same name
        index_expr[dim - event_dim] = name_to_sym(name)()

    vars_: List[Operation] = []
    for v in index_expr:
        if isinstance(v, Term):
            vars_.append(v.op)

    result = torch_getitem(tensor, tuple(index_expr))

    return result, vars_


@functools.singledispatch
def indices_of(value, **kwargs) -> IndexSet:
    """
    Get a :class:`IndexSet` of indices on which an indexed value is supported.
    :func:`indices_of` is useful in conjunction with :class:`MultiWorldCounterfactual`
    for identifying the worlds where an intervention happened upstream of a value.

    For example, in a model with an outcome variable ``Y`` and a treatment variable
    ``T`` that has been intervened on, ``T`` and ``Y`` are both indexed by ``"T"``::

        >>> def example():
        ...     with MultiWorldCounterfactual():
        ...         X = pyro.sample("X", get_X_dist())
        ...         T = pyro.sample("T", get_T_dist(X))
        ...         T = intervene(T, t, name="T_ax")  # adds an index variable "T_ax"
        ...         Y = pyro.sample("Y", get_Y_dist(X, T))
        ...         assert indices_of(X) == IndexSet()
        ...         assert indices_of(T) == IndexSet(T_ax={0, 1})
        ...         assert indices_of(Y) == IndexSet(T_ax={0, 1})
        >>> example() # doctest: +SKIP

    Just as multidimensional arrays can be expanded to shapes with new dimensions
    over which they are constant, :func:`indices_of` is defined extensionally,
    meaning that values are treated as constant functions of free variables
    not in their support.

    .. note::

        :func:`indices_of` can be extended to new value types by registering
        an implementation for the type using :func:`functools.singledispatch` .

    .. note::

        Fully general versions of :func:`indices_of` , :func:`gather`
        and :func:`scatter` would require a dependent broadcasting semantics
        for indexed values, as is the case in sparse or masked array libraries
        like ``torch.sparse`` or relational databases.

        However, this is beyond the scope of this library as it currently exists.
        Instead, :func:`gather` currently binds free variables in its input indices
        when their indices there are a strict subset of the corresponding indices
        in ``value`` , so that they no longer appear as free in the result.

        For example, in the above snippet, applying :func:`gather` to to select only
        the values of ``Y`` from worlds where no intervention on ``T`` happened
        would result in a value that no longer contains free variable ``"T"``::

            >>> indices_of(Y) == IndexSet(T_ax={0, 1}) # doctest: +SKIP
            True
            >>> Y0 = gather(Y, IndexSet(T_ax={0})) # doctest: +SKIP
            >>> indices_of(Y0) == IndexSet() != IndexSet(T_ax={0}) # doctest: +SKIP
            True

        The practical implications of this imprecision are limited
        since we rarely need to :func:`gather` along a variable twice.

    :param value: A value.
    :param kwargs: Additional keyword arguments used by specific implementations.
    :return: A :class:`IndexSet` containing the indices on which the value is supported.
    """
    raise NotImplementedError


@indices_of.register
def _indices_of_number(value: numbers.Number, **kwargs) -> IndexSet:
    return IndexSet()


@indices_of.register
def _indices_of_bool(value: bool, **kwargs) -> IndexSet:
    return IndexSet()


@indices_of.register
def _indices_of_none(value: None, **kwargs) -> IndexSet:
    return IndexSet()


@indices_of.register
def _indices_of_tuple(value: tuple, **kwargs) -> IndexSet:
    if all(isinstance(v, int) for v in value):
        return indices_of(torch.Size(value), **kwargs)
    return union(*(indices_of(v, **kwargs) for v in value))


@indices_of.register
def _indices_of_shape(value: torch.Size, **kwargs) -> IndexSet:
    name_to_dim = (
        kwargs["name_to_dim"]
        if "name_to_dim" in kwargs
        else {name: f.dim for name, f in get_index_plates().items()}
    )
    value = value[: len(value) - kwargs.get("event_dim", 0)]
    return IndexSet(
        **{
            name: set(range(value[dim]))
            for name, dim in name_to_dim.items()
            if -dim <= len(value) and value[dim] > 1
        }
    )


@indices_of.register
def _indices_of_term(value: Term, **kwargs) -> IndexSet:
    return IndexSet(
        **{
            k.__name__: set(range(v))  # type:ignore
            for (k, v) in sizesof(value).items()
        }
    )


@indices_of.register
def _indices_of_tensor(value: torch.Tensor, **kwargs) -> IndexSet:
    if isinstance(value, Term):
        return _indices_of_term(value, **kwargs)
    return indices_of(value.shape, **kwargs)


@indices_of.register
def _indices_of_distribution(
    value: pyro.distributions.torch_distribution.TorchDistributionMixin, **kwargs
) -> IndexSet:
    kwargs.pop("event_dim", None)
    return indices_of(value.batch_shape, event_dim=0, **kwargs)


@indices_of.register(dict)
def _indices_of_state(value: Dict[K, T], *, event_dim: int = 0, **kwargs) -> IndexSet:
    return union(
        *(indices_of(value[k], event_dim=event_dim, **kwargs) for k in value.keys())
    )


def gather(value: torch.Tensor, indexset: IndexSet, **kwargs) -> torch.Tensor:
    """
    Selects entries from an indexed value at the indices in a :class:`IndexSet` .
    :func:`gather` is useful in conjunction with :class:`MultiWorldCounterfactual`
    for selecting components of a value corresponding to specific counterfactual worlds.

    For example, in a model with an outcome variable ``Y`` and a treatment variable
    ``T`` that has been intervened on, we can use :func:`gather` to define quantities
    like treatment effects that require comparison of different potential outcomes::

        >>> def example():
        ...     with MultiWorldCounterfactual():
        ...         X = pyro.sample("X", get_X_dist())
        ...         T = pyro.sample("T", get_T_dist(X))
        ...         T = intervene(T, t, name="T_ax")  # adds an index variable "T_ax"
        ...         Y = pyro.sample("Y", get_Y_dist(X, T))
        ...         Y_factual = gather(Y, IndexSet(T_ax=0))         # no intervention
        ...         Y_counterfactual = gather(Y, IndexSet(T_ax=1))  # intervention
        ...         treatment_effect = Y_counterfactual - Y_factual
        >>> example() # doctest: +SKIP

    Like :func:`torch.gather` and substitution in term rewriting,
    :func:`gather` is defined extensionally, meaning that values
    are treated as constant functions of variables not in their support.

    :func:`gather` will accordingly ignore variables in ``indexset``
    that are not in the support of ``value`` computed by :func:`indices_of` .

    .. note::

        :func:`gather` can be extended to new value types by registering
        an implementation for the type using :func:`functools.singledispatch` .

    .. note::

        Fully general versions of :func:`indices_of` , :func:`gather`
        and :func:`scatter` would require a dependent broadcasting semantics
        for indexed values, as is the case in sparse or masked array libraries
        like ``scipy.sparse`` or ``xarray`` or in relational databases.

        However, this is beyond the scope of this library as it currently exists.
        Instead, :func:`gather` currently binds free variables in ``indexset``
        when their indices there are a strict subset of the corresponding indices
        in ``value`` , so that they no longer appear as free in the result.

        For example, in the above snippet, applying :func:`gather` to to select only
        the values of ``Y`` from worlds where no intervention on ``T`` happened
        would result in a value that no longer contains free variable ``"T"``::

            >>> indices_of(Y) == IndexSet(T_ax={0, 1}) # doctest: +SKIP
            True
            >>> Y0 = gather(Y, IndexSet(T_ax={0})) # doctest: +SKIP
            >>> indices_of(Y0) == IndexSet() != IndexSet(T_ax={0}) # doctest: +SKIP
            True

        The practical implications of this imprecision are limited
        since we rarely need to :func:`gather` along a variable twice.

    :param value: The value to gather.
    :param IndexSet indexset: The :class:`IndexSet` of entries to select from ``value``.
    :param kwargs: Additional keyword arguments used by specific implementations.
    :return: A new value containing entries of ``value`` from ``indexset``.
    """
    if isinstance(value, torch.Tensor) and not isinstance(value, Term):
        value, _ = lift_tensor(value, **kwargs)

    indexset_vars = {name_to_sym(name): inds for name, inds in indexset.items()}
    binding = {
        k: functools.partial(
            lambda v: v, torch_getitem(torch.tensor(list(indexset_vars[k])), [k()])
        )
        for k in sizesof(value).keys()
        if k in indexset_vars
    }

    return defun(value, *binding.keys())(*[v() for v in binding.values()])


def stack(values: Sequence[torch.Tensor], name: str, **kwargs) -> torch.Tensor:
    """Stack a sequence of indexed values, creating a new dimension. The new
    dimension is indexed by `name`. The indexed values in the stack must have
    identical shapes.

    """
    values = [v if isinstance(v, Term) else lift_tensor(v, **kwargs)[0] for v in values]
    return Indexable(torch.stack(values))[name_to_sym(name)()]


@functools.singledispatch
def cond(fst, snd, case, *, event_dim: int = 0, **kwargs):
    """
    Selection operation that is the sum-type analogue of :func:`scatter`
    in the sense that where :func:`scatter` propagates both of its arguments,
    :func:`cond` propagates only one, depending on the value of a boolean ``case`` .

    For a given ``fst`` , ``snd`` , and ``case`` , :func:`cond` returns
    ``snd`` if the ``case`` is true, and ``fst`` otherwise,
    analogous to a Python conditional expression ``snd if case else fst`` .
    Unlike a Python conditional expression, however, the case may be a tensor,
    and both branches are evaluated, as with :func:`torch.where` ::

        >>> fst, snd = torch.randn(2, 3), torch.randn(2, 3)
        >>> case = (fst < snd).all(-1)
        >>> x = cond(fst, snd, case, event_dim=1)
        >>> assert (x == torch.where(case[..., None], snd, fst)).all()

    .. note::

        :func:`cond` can be extended to new value types by registering
        an implementation for the type using :func:`functools.singledispatch` .

    :param fst: The value to return if ``case`` is ``False`` .
    :param snd: The value to return if ``case`` is ``True`` .
    :param case: A boolean value or tensor. If a tensor, should have event shape ``()`` .
    :param kwargs: Additional keyword arguments used by specific implementations.
    """
    raise NotImplementedError(f"cond not implemented for {type(fst)}")


@cond.register(int)
@cond.register(float)
@cond.register(bool)
def _cond_number(
    fst: Union[bool, numbers.Number],
    snd: Union[bool, numbers.Number, torch.Tensor],
    case: Union[bool, torch.Tensor],
    **kwargs,
) -> torch.Tensor:
    return cond(
        torch.as_tensor(fst), torch.as_tensor(snd), torch.as_tensor(case), **kwargs
    )


@cond.register
def _cond_tensor(
    fst: torch.Tensor,
    snd: torch.Tensor,
    case: torch.Tensor,
    *,
    event_dim: int = 0,
    **kwargs,
) -> torch.Tensor:
    case_ = typing.cast(
        torch.Tensor,
        case if isinstance(case, Term) else lift_tensor(case, event_dim=0, **kwargs)[0],
    )

    [fst_, snd_] = [
        v if isinstance(v, Term) else lift_tensor(v, event_dim=event_dim, **kwargs)[0]
        for v in [fst, snd]
    ]

    return torch.where(case_[(...,) + (None,) * event_dim], snd_, fst_)


def cond_n(values: Dict[IndexSet, torch.Tensor], case: torch.Tensor, **kwargs):
    assert len(values) > 0
    assert all(isinstance(k, IndexSet) for k in values.keys())
    result: Optional[torch.Tensor] = None
    for indices, value in values.items():
        tst = torch.as_tensor(
            functools.reduce(
                operator.or_, [case == index for index in next(iter(indices.values()))]
            ),
            dtype=torch.bool,
        )
        result = cond(result if result is not None else value, value, tst, **kwargs)
    return result


@pyro.poutine.runtime.effectful(type="get_index_plates")
def get_index_plates() -> (
    Dict[Hashable, pyro.poutine.indep_messenger.CondIndepStackFrame]
):
    return {}


def indexset_as_mask(
    indexset: IndexSet,
    *,
    event_dim: int = 0,
    name_to_dim_size: Optional[Dict[Hashable, Tuple[int, int]]] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Get a dense mask tensor for indexing into a tensor from an indexset.
    """
    if name_to_dim_size is None:
        name_to_dim_size = {}
        for name, f in get_index_plates().items():
            assert f.dim is not None
            name_to_dim_size[name] = (f.dim, f.size)

    batch_shape = [1] * -min([dim for dim, _ in name_to_dim_size.values()], default=0)
    inds: List[Union[slice, torch.Tensor]] = [slice(None)] * len(batch_shape)
    for name, values in indexset.items():
        dim, size = name_to_dim_size[name]
        inds[dim] = torch.tensor(list(sorted(values)), dtype=torch.long)
        batch_shape[dim] = size
    mask = torch.zeros(tuple(batch_shape), dtype=torch.bool, device=device)
    mask[tuple(inds)] = True
    return mask[(...,) + (None,) * event_dim]


def to_tensor(t: Expr[torch.Tensor], indexes=None) -> Expr[torch.Tensor]:
    return partial_eval(t, order=indexes)


Indexable = effectful.internals.sugar.Indexable
