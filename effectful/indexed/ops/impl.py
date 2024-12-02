import functools
import operator
from typing import Any, Dict, Mapping, Optional, Sequence, Set, TypeVar, Union

import torch

import effectful.indexed.internals.utils
import effectful.internals.sugar

from ...internals.sugar import partial_eval, sizesof
from ...ops.core import Expr, Operation, Term
from ...ops.function import defun

K = TypeVar("K")
T = TypeVar("T")


class IndexSet(Dict[Operation[[], int], Set[int]]):
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

    def __init__(
        self, mapping: Mapping[Operation[[], int], Union[int, Sequence[int], set[int]]]
    ):
        index_set = {}
        for k, vs in mapping.items():
            indexes = {vs} if isinstance(vs, int) else set(vs)
            if len(indexes) > 0:
                index_set[k] = indexes
        super().__init__(index_set)

    def __repr__(self):
        return f"{type(self).__name__}({super().__repr__()})"

    def __hash__(self):
        return hash(frozenset((k, frozenset(vs)) for k, vs in self.items()))

    def _to_handler(self):
        """Return an effectful handler that binds each index variable to a
        tensor of its possible index values.

        """
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
        {
            k: set.union(*[vs[k] for vs in indexsets if k in vs])
            for k in set.union(*(set(vs) for vs in indexsets))
        }
    )


def indices_of(value: Any) -> IndexSet:
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
        ...         assert indices_of(X) == IndexSet({})
        ...         assert indices_of(T) == IndexSet({T_ax: {0, 1}})
        ...         assert indices_of(Y) == IndexSet({T_ax: {0, 1}})
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
    if isinstance(value, Term):
        return IndexSet(
            {
                k: set(range(v))  # type:ignore
                for (k, v) in sizesof(value).items()
            }
        )
    elif isinstance(value, torch.distributions.Distribution):
        return indices_of(value.sample())

    return IndexSet({})


def gather(value: torch.Tensor, indexset: IndexSet) -> torch.Tensor:
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
    :return: A new value containing entries of ``value`` from ``indexset``.
    """
    binding = {
        k: functools.partial(
            lambda v: v, Indexable(torch.tensor(list(indexset[k])))[k()]
        )
        for k in sizesof(value).keys()
        if k in indexset
    }

    return defun(value, *binding.keys())(*[v() for v in binding.values()])


def stack(
    values: Union[tuple[torch.Tensor, ...], list[torch.Tensor]], dim: Operation[[], int]
) -> torch.Tensor:
    """Stack a sequence of indexed values, creating a new dimension. The new
    dimension is indexed by `dim`. The indexed values in the stack must have
    identical shapes.

    """
    return Indexable(torch.stack(values))[dim()]


def cond(fst: torch.Tensor, snd: torch.Tensor, case_: torch.Tensor) -> torch.Tensor:
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
    """
    return torch.where(
        case_.reshape(case_.shape + (1,) * min(len(snd.shape), len(fst.shape))),
        snd,
        fst,
    )


def cond_n(values: Dict[IndexSet, torch.Tensor], case: torch.Tensor) -> torch.Tensor:
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
        result = cond(result if result is not None else value, value, tst)
    assert result is not None
    return result


def to_tensor(t: Expr[torch.Tensor], indexes=None) -> Expr[torch.Tensor]:
    return partial_eval(t, order=indexes)


Indexable = effectful.internals.sugar.Indexable
