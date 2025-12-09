import collections.abc
import contextlib
import dataclasses
import types
import typing
from collections.abc import Callable
from typing import Any

from effectful.ops.syntax import _CustomSingleDispatchCallable, defop
from effectful.ops.types import (
    Expr,
    Interpretation,
    NotHandled,  # noqa: F401
    Operation,
    Term,
)

apply = Operation.__apply__


@defop
def fwd(*args, **kwargs) -> Any:
    """Forward execution to the next most enclosing handler.

    :func:`fwd` should only be called in the context of a handler.

    :param args: Positional arguments.
    :param kwargs: Keyword arguments.

    If no positional or keyword arguments are provided, :func:`fwd` will forward
    the current arguments to the next handler.

    """
    raise RuntimeError("fwd should only be called in the context of a handler")


def coproduct(intp: Interpretation, intp2: Interpretation) -> Interpretation:
    """The coproduct of two interpretations handles any effect that is handled
    by either. If both interpretations handle an effect, ``intp2`` takes
    precedence.

    Handlers in ``intp2`` that override a handler in ``intp`` may call the
    overridden handler using :func:`fwd`. This allows handlers to be written
    that extend or wrap other handlers.

    **Example usage**:

    The ``message`` effect produces a welcome message using two helper effects:
    ``greeting`` and ``name``. By handling these helper effects, we can customize the
    message.

    >>> message, greeting, name = defop(str), defop(str), defop(str)
    >>> i1 = {message: lambda: f"{greeting()} {name()}!", greeting: lambda: "Hi"}
    >>> i2 = {name: lambda: "Jack"}

    The coproduct of ``i1`` and ``i2`` handles all three effects.

    >>> i3 = coproduct(i1, i2)
    >>> with handler(i3):
    ...     print(f'{message()}')
    Hi Jack!

    We can delegate to an enclosing handler by calling :func:`fwd`. Here we
    override the ``name`` handler to format the name differently.

    >>> i4 = coproduct(i3, {name: lambda: f'*{fwd()}*'})
    >>> with handler(i4):
    ...     print(f'{message()}')
    Hi *Jack*!

    .. note::

      :func:`coproduct` allows effects to be overridden in a pervasive way, but
      this is not always desirable. In particular, an interpretation with
      handlers that call "internal" private effects may be broken if coproducted
      with an interpretation that handles those effects. It is dangerous to take
      the coproduct of arbitrary interpretations. For an alternate form of
      interpretation composition, see :func:`product`.

    """
    from effectful.internals.runtime import (
        _get_args,
        _restore_args,
        _save_args,
        _set_prompt,
    )

    res = dict(intp)
    for op, i2 in intp2.items():
        if op is fwd or op is _get_args:
            res[op] = i2  # fast path for special cases, should be equivalent if removed
        else:
            i1 = intp.get(op, op.__default_rule__)

            # calling fwd in the right handler should dispatch to the left handler
            res[op] = _set_prompt(fwd, _restore_args(_save_args(i1)), _save_args(i2))

    return res


def product(intp: Interpretation, intp2: Interpretation) -> Interpretation:
    """The product of two interpretations handles any effect that is handled by
    ``intp2``. Handlers in ``intp2`` may override handlers in ``intp``, but
    those changes are not visible to the handlers in ``intp``. In this way,
    ``intp`` is isolated from ``intp2``.

    **Example usage**:

    In this example, ``i1`` has a ``param`` effect that defines some hyperparameter and
    an effect ``f1`` that uses it. ``i2`` redefines ``param`` and uses it in a new effect
    ``f2``, which calls ``f1``.

    >>> param, f1, f2 = defop(int), defop(dict), defop(dict)
    >>> i1 = {param: lambda: 1, f1: lambda: {'inner': param()}}
    >>> i2 = {param: lambda: 2, f2: lambda: f1() | {'outer': param()}}

    Using :func:`product`, ``i2``'s override of ``param`` is not visible to ``i1``.

    >>> with handler(product(i1, i2)):
    ...     print(f2())
    {'inner': 1, 'outer': 2}

    However, if we use :func:`coproduct`, ``i1`` is not isolated from ``i2``.

    >>> with handler(coproduct(i1, i2)):
    ...     print(f2())
    {'inner': 2, 'outer': 2}

    **References**

    [1] Ahman, D., & Bauer, A. (2020, April). Runners in action. In European
    Symposium on Programming (pp. 29-55). Cham: Springer International
    Publishing.

    """
    if any(op in intp for op in intp2):  # alpha-rename
        renaming: Interpretation = {op: defop(op) for op in intp2 if op in intp}
        intp_fresh = {renaming.get(op, op): handler(renaming)(intp[op]) for op in intp}
        return product(intp_fresh, intp2)
    else:
        refls2 = {op: op.__default_rule__ for op in intp2}
        intp_ = coproduct({}, {op: runner(refls2)(intp[op]) for op in intp})
        return {op: runner(intp_)(intp2[op]) for op in intp2}


@contextlib.contextmanager
def runner(intp: Interpretation):
    """Install an interpretation by taking a product with the current
    interpretation.

    """
    from effectful.internals.runtime import get_interpretation, interpreter

    @interpreter(get_interpretation())
    def _reapply[**P, S](op: Operation[P, S], *args: P.args, **kwargs: P.kwargs):
        return op(*args, **kwargs)

    with interpreter({apply: _reapply, **intp}):
        yield intp


@contextlib.contextmanager
def handler(intp: Interpretation):
    """Install an interpretation by taking a coproduct with the current
    interpretation.

    """
    from effectful.internals.runtime import get_interpretation, interpreter

    with interpreter(coproduct(get_interpretation(), intp)):
        yield intp


@_CustomSingleDispatchCallable
def evaluate[T](
    __dispatch: Callable[[type], Callable[..., Expr[T]]],
    expr: Expr[T],
    *,
    intp: Interpretation | None = None,
) -> Expr[T]:
    """Evaluate expression ``expr`` using interpretation ``intp``. If no
    interpretation is provided, uses the current interpretation.

    :param expr: The expression to evaluate.
    :param intp: Optional interpretation for evaluating ``expr``.

    **Example usage**:

    >>> @defop
    ... def add(x: int, y: int) -> int:
    ...     raise NotHandled
    >>> expr = add(1, add(2, 3))
    >>> print(str(expr))
    add(1, add(2, 3))
    >>> evaluate(expr, intp={add: lambda x, y: x + y})
    6

    """
    from effectful.internals.runtime import interpreter

    if intp is not None:
        return interpreter(intp)(evaluate)(expr)

    return __dispatch(type(expr))(expr)


@evaluate.register(object)
@evaluate.register(str)
@evaluate.register(bytes)
def _evaluate_object[T](expr: T, **kwargs) -> T:
    if dataclasses.is_dataclass(expr) and not isinstance(expr, type):
        return typing.cast(
            T,
            dataclasses.replace(
                expr,
                **{
                    field.name: evaluate(getattr(expr, field.name))
                    for field in dataclasses.fields(expr)
                },
            ),
        )
    return expr


@evaluate.register(Term)
def _evaluate_term(expr: Term, **kwargs):
    args = tuple(evaluate(arg) for arg in expr.args)
    kwargs = {k: evaluate(v) for k, v in expr.kwargs.items()}
    return expr.op(*args, **kwargs)


@evaluate.register(Operation)
def _evaluate_operation(expr: Operation, **kwargs) -> Operation:
    from effectful.internals.runtime import get_interpretation

    op_intp = get_interpretation().get(expr, expr)
    return op_intp if isinstance(op_intp, Operation) else expr


@evaluate.register(collections.defaultdict)
def _evaluate_defaultdict(expr, **kwargs):
    return type(expr)(expr.default_factory, evaluate(tuple(expr.items())))


@evaluate.register(types.MappingProxyType)
def _evaluate_mappingproxytype(expr, **kwargs):
    return type(expr)(dict(evaluate(tuple(expr.items()))))


@evaluate.register(collections.abc.Mapping)
def _evaluate_mapping(expr, **kwargs):
    return type(expr)(evaluate(tuple(expr.items())))


@evaluate.register(tuple)
def _evaluate_tuple(expr, **kwargs):
    if (
        isinstance(expr, tuple)
        and hasattr(expr, "_fields")
        and all(hasattr(expr, field) for field in getattr(expr, "_fields"))
    ):  # namedtuple
        return type(expr)(
            **{field: evaluate(getattr(expr, field)) for field in expr._fields}
        )
    else:
        return type(expr)(evaluate(item) for item in expr)


@evaluate.register(collections.abc.Sequence)
def _evaluate_sequence(expr, **kwargs):
    return type(expr)(evaluate(item) for item in expr)


@evaluate.register(collections.abc.ItemsView)
@evaluate.register(collections.abc.KeysView)
def _evaluate_set_view(expr, **kwargs):
    return {evaluate(item) for item in expr}


@evaluate.register(collections.abc.ValuesView)
def _evaluate_list_view(expr, **kwargs):
    return [evaluate(item) for item in expr]


def _simple_type(tp: type) -> type:
    """Convert a type object into a type that can be dispatched on."""
    if isinstance(tp, typing.TypeVar):
        tp = (
            tp.__bound__
            if tp.__bound__
            else tp.__constraints__[0]
            if tp.__constraints__
            else object
        )
    if isinstance(tp, types.UnionType):
        raise TypeError(f"Union types are not supported: {tp}")
    return typing.get_origin(tp) or tp


def typeof[T](term: Expr[T]) -> type[T]:
    """Return the type of an expression.

    **Example usage**:

    Type signatures are used to infer the types of expressions.

    >>> @defop
    ... def cmp(x: int, y: int) -> bool:
    ...     raise NotHandled
    >>> typeof(cmp(1, 2))
    <class 'bool'>

    Types can be computed in the presence of type variables.

    >>> @defop
    ... def if_then_else[T](x: bool, a: T, b: T) -> T:
    ...     raise NotHandled
    >>> typeof(if_then_else(True, 0, 1))
    <class 'int'>

    """
    from effectful.internals.runtime import interpreter
    from effectful.internals.unification import Box

    def _apply(op, *args, **kwargs):
        return Box(op.__type_rule__(*args, **kwargs))

    with interpreter({apply: _apply}):
        type_or_value = evaluate(term)
        if isinstance(type_or_value, Box):
            return _simple_type(type_or_value.value)
        return typing.cast(type[T], type(type_or_value))


def fvsof[S](term: Expr[S]) -> collections.abc.Set[Operation]:
    """Return the free variables of an expression.

    **Example usage**:

    >>> @defop
    ... def f(x: int, y: int) -> int:
    ...     raise NotHandled
    >>> fvs = fvsof(f(1, 2))
    >>> assert f in fvs
    >>> assert len(fvs) == 1
    """
    from effectful.internals.runtime import interpreter

    _fvs: set[Operation] = set()

    def _update_fvs(op, *args, **kwargs):
        _fvs.add(op)
        bindings = op.__fvs_rule__(*args, **kwargs)
        for bound_var in set().union(*(*bindings.args, *bindings.kwargs.values())):
            assert isinstance(bound_var, Operation)
            if bound_var in _fvs:
                _fvs.remove(bound_var)

    with interpreter({apply: _update_fvs}):
        evaluate(term)

    return _fvs
