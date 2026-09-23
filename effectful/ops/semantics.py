import collections.abc
import contextlib
import dataclasses
import functools
import inspect
import operator
import types
import typing
import weakref

from effectful.internals.runtime import cache
from effectful.internals.weak import weak_memoize
from effectful.ops.syntax import (
    ConstructorOperation,
    DataclassConstructorOperation,
    ObjectInterpretation,
    Scoped,
    _CustomSingleDispatchCallable,
    _Renaming,
    defop,
    implements,
)
from effectful.ops.types import (
    Expr,
    Interpretation,
    NotHandled,  # noqa: F401
    Operation,
    Term,
)

if typing.TYPE_CHECKING:
    import typing_extensions


apply = Operation.__apply__


@defop
def fwd(*args, **kwargs) -> typing.Any:
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
        _save_args,
        _save_then_restore_args,
        _set_prompt,
    )

    res = dict(intp)
    for op, i2 in intp2.items():
        if op in {fwd, _get_args}:
            res[op] = i2  # fast path for special cases, should be equivalent if removed
        else:
            # calling fwd in the right handler should dispatch to the left handler
            i1 = intp.get(op)
            res[op] = (
                _set_prompt(fwd, _save_then_restore_args(i1), _save_args(i2))
                if i1 is not None
                else _save_args(i2)
            )

    return res


@contextlib.contextmanager
def handler(intp: Interpretation):
    """Install an interpretation by taking a coproduct with the current
    interpretation.

    """
    from effectful.internals.runtime import get_interpretation, interpreter

    with interpreter(coproduct(get_interpretation(), intp)):
        yield intp


@ConstructorOperation.define
def as_tuple(*args) -> tuple:
    return tuple(args)


_MISSING: typing.Any = object()


@_CustomSingleDispatchCallable
def evaluate[T](
    __dispatch: collections.abc.Callable[
        [type], collections.abc.Callable[..., Expr[T]]
    ],
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
    from effectful.internals.runtime import (
        EVAL_CACHE,
        cache_get,
        cache_put,
        get_interpretation,
        interpreter,
    )

    with interpreter(intp if intp is not None else get_interpretation()) as current:
        store = EVAL_CACHE.get()
        if store is None:
            # No cache installed. Open one for the duration of this call and start
            # over. Without it, an expression that reaches a subexpression along
            # several paths re-evaluates it once per path, which is exponential in
            # the depth of a DAG. Only the outermost call takes this branch, so the
            # extra re-entry is paid once.
            with cache():
                return evaluate(expr, intp=current)

        result = cache_get(store, expr, current, _MISSING)
        if result is not _MISSING:
            return result

        built = __dispatch(type(expr))(expr)
        result = expr if _is_rebuild(built, expr) else built
        if isinstance(expr, Term):
            for node in (built, expr):
                if getattr(node, "__unrenamed_term__", None) is not None:
                    object.__delattr__(node, "__unrenamed_term__")

        cache_put(store, expr, current, result)
        return result


def _same_node(a: typing.Any, b: Term) -> bool:
    """Whether two terms apply the same operation to the very same operands.

    ``a`` is whatever a rule produced, or the record of what a node was built from,
    which may be absent -- so it is checked rather than assumed to be a term.

    Operands are compared by identity, which is exact rather than approximate because
    :func:`evaluate` hands back the object it was given wherever nothing changed: a
    rebuilt operand is a new object precisely when it differs.
    """
    return (
        isinstance(a, Term)
        and isinstance(b, Term)
        and a.op is b.op
        and len(a.args) == len(b.args)
        and a.kwargs.keys() == b.kwargs.keys()
        and all(x is y for x, y in zip(a.args, b.args))
        and all(v is b.kwargs[k] for k, v in a.kwargs.items())
    )


def _is_rebuild(result: typing.Any, expr: typing.Any) -> bool:
    """Whether ``result`` is a copy of ``expr`` assembled from the very same parts.
    The comparison is one level deep and by identity, which is exact rather than
    approximate: evaluating a part already returns the object it was given when that
    part did not change, so parts differ precisely when they are not the same object.
    """
    if result is expr:
        return True
    elif type(result) is not type(expr):
        return False
    elif isinstance(expr, Term):
        return _same_node(result, expr) or _same_node(
            getattr(result, "__unrenamed_term__", None), expr
        )
    elif isinstance(expr, str | bytes):
        return False
    elif dataclasses.is_dataclass(expr):
        names = [f.name for f in dataclasses.fields(expr)]
        return _is_rebuild(
            {name: getattr(result, name) for name in names},
            {name: getattr(expr, name) for name in names},
        )
    elif isinstance(expr, collections.abc.Mapping):
        return {(id(k), id(v)) for k, v in result.items()} == {
            (id(k), id(v)) for k, v in expr.items()
        }
    elif isinstance(expr, collections.abc.Set):
        return {id(v) for v in result} == {id(v) for v in expr}
    elif isinstance(expr, collections.abc.Sequence):
        parts, originals = list(iter(result)), list(iter(expr))
        return len(parts) == len(originals) and all(
            a is b for a, b in zip(parts, originals)
        )
    else:
        return False


@evaluate.register(object)
@evaluate.register(str)
@evaluate.register(bytes)
def _evaluate_object[T](expr: T, **kwargs) -> T:
    if dataclasses.is_dataclass(expr) and not isinstance(expr, type):
        return _evaluate_dataclass(expr, **kwargs)
    return expr


def _evaluate_dataclass[T](expr: T, **kwargs) -> T:
    subst = {
        field.name: evaluate(getattr(expr, field.name))
        for field in dataclasses.fields(expr)  # type: ignore[arg-type]
    }
    return typing.cast(
        T,
        DataclassConstructorOperation.define(type(expr))(**subst),  # type: ignore[arg-type]
    )


@weak_memoize(cache=weakref.WeakKeyDictionary())
def _binds_vars(op: Operation) -> bool:
    """Whether ``op`` binds variables in any of its operands.

    True when some parameter's scope is not contained in the return value's, the
    condition :meth:`Scoped.analyze` uses to decide that a parameter contributes bound
    variables. Memoized per operation, since it is a property of the signature and
    :func:`_evaluate_term` asks it of every node it visits.
    """
    sig = op._signature_with_scopes
    returned = Scoped._get_param_ordinal(sig.return_annotation)
    return any(
        not (Scoped._get_param_ordinal(p) <= returned) for p in sig.parameters.values()
    )


class _Shadow(dict):
    """An interpretation in which some variables are bound, so handled as unhandled.

    Remembers what it shadows and what it shadows it over, so that :func:`evaluate`
    can drop back to ``parent`` for a term none of ``bound`` can occur in.
    """

    parent: Interpretation
    bound: frozenset[Operation]


def _shadowed(intp: Interpretation, bound) -> Interpretation:
    shadow = _Shadow(
        coproduct(intp, {b: functools.partial(b.__apply__, b) for b in bound})
    )
    shadow.parent, shadow.bound = intp, frozenset(bound)
    return shadow


@evaluate.register(Term)
def _evaluate_term(expr: Term, **kwargs):
    from effectful.internals.runtime import RECONSTRUCTING, get_interpretation

    intp = get_interpretation()

    # Renaming is the identity on a term that mentions none of the variables being
    # renamed, so such a term already stands for its own rebuild.
    if isinstance(intp, _Renaming) and not (fvsof(expr) & intp.vars):
        return expr

    # Likewise a shadow is the identity on a term none of its variables can occur in.
    # Dropping back to the parent is not just equivalent but necessary: evaluation is
    # memoized per interpretation, so a term below a binder would otherwise be
    # recomputed once for each distinct shadow it is reached under.
    if isinstance(intp, _Shadow) and not (fvsof(expr) & intp.bound):
        return evaluate(expr, intp=intp.parent)

    binds_vars = _binds_vars(expr.op)
    if binds_vars:
        # A variable bound in an operand is re-entered there as an unhandled
        # operation, so that a substitution for it stops at this binder instead of
        # reaching inside it. The bound set differs per operand: ``Let`` binds its
        # variable in the body but not in the value.
        bindings = expr.op.__fvs_rule__(*expr.args, **expr.kwargs)
        args = tuple(
            evaluate(arg, intp=_shadowed(intp, bound))
            if bound and any(b in intp for b in bound)
            else evaluate(arg)
            for arg, bound in zip(expr.args, bindings.args, strict=True)
        )
        kwargs = {
            k: evaluate(v, intp=_shadowed(intp, bindings.kwargs[k]))
            if bindings.kwargs[k] and any(b in intp for b in bindings.kwargs[k])
            else evaluate(v)
            for k, v in expr.kwargs.items()
        }
    else:
        args = tuple(evaluate(arg) for arg in expr.args)
        kwargs = {k: evaluate(v) for k, v in expr.kwargs.items()}

    # set context for fast path in defdata to avoid rename-then-discard
    token = RECONSTRUCTING.set(expr)
    try:
        return expr.op(*args, **kwargs)
    finally:
        RECONSTRUCTING.reset(token)


@evaluate.register(Operation)
def _evaluate_operation(expr: Operation, **kwargs) -> Operation:
    from effectful.internals.runtime import get_interpretation

    op_intp = get_interpretation().get(expr, expr)
    return op_intp if isinstance(op_intp, Operation) else expr


@evaluate.register(collections.defaultdict)
def _evaluate_defaultdict(expr, **kwargs):
    return ConstructorOperation.define(type(expr))(
        expr.default_factory,
        as_tuple(*(evaluate(item) for item in expr.items())),
    )


@evaluate.register(types.MappingProxyType)
def _evaluate_mappingproxytype(expr, **kwargs):
    return ConstructorOperation.define(type(expr))(
        as_tuple(*(evaluate(item) for item in expr.items()))
    )


@evaluate.register(collections.abc.Mapping)
def _evaluate_mapping(expr, **kwargs):
    return ConstructorOperation.define(type(expr))(
        as_tuple(*(evaluate(item) for item in expr.items()))
    )


@evaluate.register(tuple)
def _evaluate_tuple(expr, **kwargs):
    if (
        isinstance(expr, tuple)
        and hasattr(expr, "_fields")
        and all(hasattr(expr, field) for field in getattr(expr, "_fields"))
    ):  # namedtuple
        return ConstructorOperation.define(type(expr))(
            **{field: evaluate(getattr(expr, field)) for field in expr._fields}
        )
    else:
        return ConstructorOperation.define(type(expr))(
            as_tuple(*(evaluate(item) for item in expr))
        )


@evaluate.register(collections.abc.Sequence)
def _evaluate_sequence(expr, **kwargs):
    return ConstructorOperation.define(type(expr))(
        as_tuple(*(evaluate(item) for item in expr))
    )


@evaluate.register(collections.abc.ItemsView)
@evaluate.register(collections.abc.KeysView)
def _evaluate_set_view(expr, **kwargs):
    return ConstructorOperation.define(set)(
        as_tuple(*(evaluate(item) for item in expr))
    )


@evaluate.register(collections.abc.ValuesView)
def _evaluate_list_view(expr, **kwargs):
    return ConstructorOperation.define(list)(
        as_tuple(*(evaluate(item) for item in expr))
    )


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
    if typing.get_origin(tp) == typing.Literal:
        args = typing.get_args(tp)
        if not args:
            raise TypeError(
                "Literal annotations must be supplied with at least one argument"
            )
        tp = functools.reduce(operator.or_, (type(arg) for arg in args))
    if isinstance(tp, types.UnionType):
        raise TypeError(f"Union types are not supported: {tp}")
    return typing.get_origin(tp) or tp


class _TypeofIntp(ObjectInterpretation):
    @implements(apply)
    def _apply(self, op, *args, **kwargs):
        from effectful.internals.unification import Box

        return Box(op.__type_rule__(*args, **kwargs))

    @implements(ConstructorOperation.__apply__)
    def _constructor_apply(self, op, *args, **kwargs):
        return op.__default_rule__(*args, **kwargs)

    @implements(DataclassConstructorOperation.__apply__)
    def _dataclass_constructor_apply(self, op, *args, **kwargs):
        from effectful.internals.unification import Box

        return Box(op.__type_rule__(*args, **kwargs))


_TYPEOF_INTP = _TypeofIntp()


@typing.overload
def typeof[T](
    term: Expr[T], *, keep_params: typing.Literal[False] = False
) -> type[T]: ...


@typing.overload
def typeof[T](
    term: Expr[T], *, keep_params: typing.Literal[True]
) -> "typing_extensions.TypeForm[T]": ...


def typeof[T](term: Expr[T], *, keep_params: bool = False) -> typing.Any:
    """Return the type of an expression.

    :param keep_params: Return the full inferred type rather than a class to
        dispatch on, keeping type parameters that would otherwise be dropped.

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

    By default the result is simplified to a class, which is what dispatching
    on a term's type needs. ``keep_params`` returns the parameters as well:

    >>> @defop
    ... def duplicate[T](x: T) -> list[T]:
    ...     raise NotHandled
    >>> typeof(duplicate("a"))
    <class 'list'>
    >>> typeof(duplicate("a"), keep_params=True)
    list[str]

    """
    from effectful.internals.unification import Box, nested_type

    if isinstance(term, Term) and (constant := _constant_type(term.op)) is not None:
        from effectful.internals.runtime import EVAL_CACHE, cache_put

        boxed = Box(constant)
        store = EVAL_CACHE.get()
        if store is not None:
            cache_put(store, term, _TYPEOF_INTP, boxed)
        if keep_params:
            return typing.cast(type[T], nested_type(boxed).value)
        return typing.cast(type[T], _simple_type(constant))

    type_or_value = evaluate(term, intp=_TYPEOF_INTP)
    if not keep_params and isinstance(type_or_value, Box):
        return _simple_type(type_or_value.value)
    elif not keep_params:
        return typing.cast(type[T], type(type_or_value))
    else:
        return typing.cast(type[T], nested_type(type_or_value).value)


@weak_memoize(cache=weakref.WeakKeyDictionary())
def _constant_type(op: Operation) -> type | None:
    """The analysis of every node of ``op``, where the operands cannot change it.

    :meth:`Operation.__type_rule__` hands back the return annotation untouched unless it
    has free type variables, so for most operations the answer is in the signature and
    the operands need not be analyzed at all. ``None`` where they do.

    The annotation as written, not what :func:`typeof` reduces it to: a node above this
    one unifies against the full type, so ``list[int]`` may not arrive there as ``list``.
    """
    from effectful.internals.unification import freetypevars

    if isinstance(op, ConstructorOperation):
        # Not analyzed through ``apply``, so the rule above is not the one that applies.
        return None

    anno = op.__signature__.return_annotation
    if typing.get_origin(anno) is typing.Annotated:
        anno = typing.get_args(anno)[0]

    if anno is inspect.Parameter.empty:
        return object
    elif anno is None:
        return type(None)
    elif freetypevars(anno):
        return None

    try:
        _simple_type(anno)  # a union annotation, which _simple_type rejects
    except TypeError:
        return None
    return anno


class _FvsAnalysis(typing.NamedTuple):
    ops: frozenset[Operation] = frozenset()
    fvs: frozenset[Operation] = frozenset()


class _FvsofIntp(ObjectInterpretation):
    @staticmethod
    def _analysis(value) -> _FvsAnalysis:
        if isinstance(value, _FvsAnalysis):
            return value
        else:
            return _FvsAnalysis(Scoped.extract_operations(value))

    @implements(ConstructorOperation.__apply__)
    def _apply_collection_binders(self, op, *args, **kwargs):
        analyses = tuple(self._analysis(x) for x in (*args, *kwargs.values()))
        return _FvsAnalysis(
            frozenset().union(frozenset(), *(a.ops for a in analyses)),
            frozenset().union(frozenset(), *(a.fvs for a in analyses)),
        )

    @implements(apply)
    def _apply_fvs(self, op, *args, **kwargs):
        arg_analyses = tuple(self._analysis(a) for a in args)
        kwarg_analyses = {k: self._analysis(v) for k, v in kwargs.items()}
        bindings = op.__fvs_rule__(
            *(a.ops for a in arg_analyses),
            **{k: a.ops for k, a in kwarg_analyses.items()},
        )
        binders = frozenset().union(*(*bindings.args, *bindings.kwargs.values()))
        fvs = frozenset().union(
            {op}, *(a.fvs for a in (*arg_analyses, *kwarg_analyses.values()))
        )
        return _FvsAnalysis(fvs=fvs - binders)


_FVSOF_INTP = _FvsofIntp()


def fvsof[S](term: Expr[S]) -> collections.abc.Set[Operation]:
    """Return the free operations in a term.

    An operation belongs to `fvsof(t)` when it appears free in the term `t`.
    This excludes operations like `apply` or collection constructors that are
    raised during `evaluate` but do not appear in `t`. It also excludes
    operations that are bound by a `Scoped` operation. However, it is not
    restricted to the nullary operations in `t`.

    **Example usage**:

    `fvsof` includes all unbound operations in a term:

    >>> a = defop(int)
    >>> @defop
    ... def f(x: int, y: int) -> int:
    ...     raise NotHandled
    >>> fvs = fvsof(f(a(), 1))
    >>> assert fvs >= {f, a}

    `fvsof` accepts the same values as `evaluate`, including collections:

    >>> fvs = fvsof([a(), {'k': f(0, 1)}])
    >>> assert fvs >= {f, a}

    """
    result = evaluate(term, intp=_FVSOF_INTP)
    return result.fvs if isinstance(result, _FvsAnalysis) else frozenset()
