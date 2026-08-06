import collections.abc
import functools
import types
import typing

from effectful.ops.semantics import ConstructorOperation, apply, evaluate
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Operation

type IndexElement[T] = (
    None | int | slice | collections.abc.Sequence[int] | types.EllipsisType | T
)


def _desugar_tensor_index[T](
    shape: tuple[int, ...], key: collections.abc.Sequence[IndexElement[T]]
) -> tuple[tuple[int, ...], tuple[IndexElement[T], ...]]:
    new_shape: list[int] = []
    new_key: list[IndexElement[T]] = []

    def extra_dims(key: collections.abc.Sequence[IndexElement[T]]) -> int:
        return sum(1 for k in key if k is None)

    # handle any missing dimensions by adding a trailing Ellipsis
    if not any(k is Ellipsis for k in key):
        key = tuple(key) + (...,)

    for i, k in enumerate(key):
        if k is None:  # add a new singleton dimension
            new_shape.append(1)
            new_key.append(slice(None))
        elif k is Ellipsis:
            assert not any(k is Ellipsis for k in key[i + 1 :]), (
                "only one Ellipsis allowed"
            )

            # determine which of the original dimensions this ellipsis refers to
            pre_dims = i - extra_dims(key[:i])  # dimensions that precede the ellipsis
            elided_dims = (
                len(shape) - pre_dims - (len(key) - i - 1 - extra_dims(key[i + 1 :]))
            )  #
            new_shape += shape[pre_dims : pre_dims + elided_dims]
            new_key += [slice(None)] * elided_dims
        else:
            new_shape.append(shape[len(new_shape) - extra_dims(key[:i])])
            new_key.append(k)

    return tuple(new_shape), tuple(new_key)


class _Name[T]:
    """An index entry that names a dimension: a bare call to ``op``.

    Deliberately not a tuple, so that a key can be told apart from an entry.
    """

    __slots__ = ("op",)

    def __init__(self, op: Operation[[], T]):
        self.op = op


#: An index entry that is a term but not a bare name, so it neither names a
#: dimension nor leaves the indexed result with a shape this analysis can
#: predict. Distinct from a concrete entry, which does neither but is harmless.
_OPAQUE: typing.Any = object()


class _SizeAnalysis[T](typing.NamedTuple):
    """What the analysis of a single node carries.

    ``sizes`` is the result. The rest is what a parent `__getitem__`
    needs to finish its own analysis, which the sizes alone cannot supply:
    ``shape`` when the node denotes an array whose shape is known, and
    ``index`` for what the node looks like in a key -- the dimension it names,
    or the value it already is.
    """

    sizes: dict[Operation[[], T], int]
    index: typing.Any
    shape: tuple[int, ...] | None = None


class _BaseSizesofIntp[T](ObjectInterpretation):
    """Shared part of the analysis behind ``sizesof``.

    A backend supplies the array type it analyses and wires its own getitem
    operation to :meth:`_getitem`; the analysis itself is common to all of
    them. What it decides about a getitem -- which entries name a dimension,
    and whether the result is eager and so has a shape at all -- has to match
    what that backend's :func:`defdata` rule decides, or the analysis will
    predict a shape for a term that is never built with one, or miss one that
    is.
    """

    arr_type: typing.ClassVar[type] = object

    @classmethod
    def _analysis(cls, value) -> _SizeAnalysis[T]:
        """View a rule argument as an analysis. Leaves contribute no sizes.

        A leaf stands for itself in a key, so that keys rebuild into real
        tuples holding real slices and ``None`` and ``Ellipsis`` literals.
        """
        if isinstance(value, _SizeAnalysis):
            return value
        elif isinstance(value, cls.arr_type):
            return _SizeAnalysis[T]({}, value, value.shape)  # type: ignore
        else:
            return _SizeAnalysis[T]({}, value)

    @staticmethod
    def _merge(
        s1: dict[Operation[[], T], int], s2: dict[Operation[[], T], int]
    ) -> dict[Operation[[], T], int]:
        s3 = s1.copy()
        for k, v in s2.items():
            if k in s3 and s3[k] != v:
                raise ValueError(
                    f"Named index {k} used in incompatible dimensions of size {s3[k]} and {v}"
                )
            s3[k] = v
        return s3

    @implements(apply)
    def _apply(self, op, *args, **kwargs):
        analyses = tuple(self._analysis(x) for x in (*args, *kwargs.values()))
        return _SizeAnalysis(
            functools.reduce(self._merge, (a.sizes for a in analyses), {}),
            # Only a bare call can name a dimension, which is the test each
            # backend's ``defdata`` rule indexes eagerly under.
            _Name(op) if not (args or kwargs) else _OPAQUE,
        )

    @implements(ConstructorOperation.__apply__)
    def _apply_constructor(self, op, *args, **kwargs):
        arg_analyses = tuple(self._analysis(x) for x in args)
        kwarg_analyses = {k: self._analysis(v) for k, v in kwargs.items()}
        analyses = (*arg_analyses, *kwarg_analyses.values())
        return _SizeAnalysis(
            functools.reduce(self._merge, (a.sizes for a in analyses), {}),
            op.__default_rule__(
                *(a.index for a in arg_analyses),
                **{k: a.index for k, a in kwarg_analyses.items()},
            ),
        )

    def _getitem(self, x, key):
        is_concrete = isinstance(x, self.arr_type)
        x, key = self._analysis(x), self._analysis(key)
        sizes = self._merge(x.sizes, key.sizes)

        if x.shape is None or not isinstance(key.index, tuple | list):
            return _SizeAnalysis(sizes, _OPAQUE)

        shape, entries = _desugar_tensor_index(x.shape, key.index)
        for i, entry in enumerate(entries):
            if isinstance(entry, _Name):
                sizes = self._merge(sizes, {entry.op: shape[i]})

        eager = is_concrete and not any(e is _OPAQUE for e in entries)
        return _SizeAnalysis(
            sizes,
            _OPAQUE,
            tuple(s for s, e in zip(shape, entries) if not isinstance(e, _Name))
            if eager
            else None,
        )


def _sizesof[T](
    value, *, analysis: _BaseSizesofIntp[T]
) -> collections.abc.Mapping[Operation[[], T], int]:
    """Return a mapping from named dimensions to their sizes.

    Raises a ValueError if the same name is used for different sizes.
    """
    result = evaluate(value, intp=analysis)
    return result.sizes if isinstance(result, _SizeAnalysis) else {}
