import contextlib
import functools
from typing import Any, Callable, Optional, Set, Type, TypeVar

import tree
from typing_extensions import ParamSpec

from effectful.ops.syntax import deffn, defop
from effectful.ops.types import Expr, Interpretation, Operation, Term

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@defop
def apply(intp: Interpretation, op: Operation, *args, **kwargs) -> Any:
    """Apply ``op`` to ``args``, ``kwargs`` in interpretation ``intp``.

    Handling :func:`apply` changes the evaluation strategy of terms.

    **Example usage**:

    >>> @defop
    ... def add(x: int, y: int) -> int:
    ...     return x + y
    >>> @defop
    ... def mul(x: int, y: int) -> int:
    ...     return x * y

    ``add`` and ``mul`` have default rules, so this term evaluates:

    >>> mul(add(1, 2), 3)
    9

    By installing an :func:`apply` handler, we capture the term instead:

    >>> with handler({apply: lambda _, op, *args, **kwargs: op.__free_rule__(*args, **kwargs) }):
    ...     term = mul(add(1, 2), 3)
    >>> term
    mul(add(1, 2), 3)

    """
    if op in intp:
        return intp[op](*args, **kwargs)
    elif apply in intp:
        return intp[apply](intp, op, *args, **kwargs)
    else:
        return op.__default_rule__(*args, **kwargs)


@defop  # type: ignore
def call(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """An operation that eliminates a callable term.

    This operation is invoked by the ``__call__`` method of a callable term.

    """
    if isinstance(fn, Term) and fn.op is deffn:
        body: Expr[Callable[P, T]] = fn.args[0]
        argvars: tuple[Operation, ...] = fn.args[1:]
        kwvars: dict[str, Operation] = fn.kwargs
        subs = {
            **{v: functools.partial(lambda x: x, a) for v, a in zip(argvars, args)},
            **{kwvars[k]: functools.partial(lambda x: x, kwargs[k]) for k in kwargs},
        }
        with handler(subs):
            return evaluate(body)
    elif not any(isinstance(a, Term) for a in tree.flatten((fn, args, kwargs))):
        return fn(*args, **kwargs)
    else:
        raise NotImplementedError


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
        renaming = {op: defop(op) for op in intp2 if op in intp}
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
    def _reapply(_, op: Operation[P, S], *args: P.args, **kwargs: P.kwargs):
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


def evaluate(expr: Expr[T], *, intp: Optional[Interpretation] = None) -> Expr[T]:
    """Evaluate expression ``expr`` using interpretation ``intp``. If no
    interpretation is provided, uses the current interpretation.

    :param expr: The expression to evaluate.
    :param intp: Optional interpretation for evaluating ``expr``.

    **Example usage**:

    >>> @defop
    ... def add(x: int, y: int) -> int:
    ...     raise NotImplementedError
    >>> expr = add(1, add(2, 3))
    >>> expr
    add(1, add(2, 3))
    >>> evaluate(expr, intp={add: lambda x, y: x + y})
    6

    """
    if intp is None:
        from effectful.internals.runtime import get_interpretation

        intp = get_interpretation()

    if isinstance(expr, Term):
        (args, kwargs) = tree.map_structure(
            functools.partial(evaluate, intp=intp), (expr.args, expr.kwargs)
        )
        return apply.__default_rule__(intp, expr.op, *args, **kwargs)
    elif tree.is_nested(expr):
        return tree.map_structure(functools.partial(evaluate, intp=intp), expr)
    else:
        return expr


def typeof(term: Expr[T]) -> Type[T]:
    """Return the type of an expression.

    **Example usage**:

    Type signatures are used to infer the types of expressions.

    >>> @defop
    ... def cmp(x: int, y: int) -> bool:
    ...     raise NotImplementedError
    >>> typeof(cmp(1, 2))
    <class 'bool'>

    Types can be computed in the presence of type variables.

    >>> from typing import TypeVar
    >>> T = TypeVar('T')
    >>> @defop
    ... def if_then_else(x: bool, a: T, b: T) -> T:
    ...     raise NotImplementedError
    >>> typeof(if_then_else(True, 0, 1))
    <class 'int'>

    """
    from effectful.internals.runtime import interpreter

    with interpreter({apply: lambda _, op, *a, **k: op.__type_rule__(*a, **k)}):
        return evaluate(term) if isinstance(term, Term) else type(term)  # type: ignore


def fvsof(term: Expr[S]) -> Set[Operation]:
    """Return the free variables of an expression.

    **Example usage**:

    >>> @defop
    ... def f(x: int, y: int) -> int:
    ...     raise NotImplementedError
    >>> fvsof(f(1, 2))
    {f}

    """
    from effectful.internals.runtime import interpreter

    _fvs: Set[Operation] = set()

    def _update_fvs(_, op, *args, **kwargs):
        _fvs.add(op)
        arg_ctxs, kwarg_ctxs = op.__fvs_rule__(*args, **kwargs)
        bound_vars = set().union(
            *(a for a in arg_ctxs),
            *(k for k in kwarg_ctxs.values()),
        )
        for bound_var in bound_vars:
            if bound_var in _fvs:
                _fvs.remove(bound_var)

    with interpreter({apply: _update_fvs}):
        evaluate(term)

    return _fvs
