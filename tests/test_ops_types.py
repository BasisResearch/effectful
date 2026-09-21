import dataclasses
import inspect
import typing

from effectful.ops.semantics import typeof
from effectful.ops.syntax import defop
from effectful.ops.types import Interpretation, NotHandled


def test_prettyprinter_traverses_dataclasses_and_raw_operations():
    prettyprinter = __import__("prettyprinter")

    @dataclasses.dataclass
    class Result:
        value: object
        assignment: object

    x = defop(int, name="x")

    @defop
    def reduce(body: object, streams: object) -> object:
        raise NotHandled

    expr = reduce(Result((x() - 1) ** 2, {x: x()}), {x: (0, 1, 2)})

    formatted = prettyprinter.pformat(expr, width=80)

    assert "value=(x() - 1) ** 2" in formatted
    assert "assignment={x: x()}" in formatted
    assert "{x: (0, 1, 2)}" in formatted
    assert "Operation(" not in formatted
    assert "_IntegralTerm(" not in formatted


def test_prettyprinter_prints_raw_operation_as_its_name():
    prettyprinter = __import__("prettyprinter")
    x = defop(int, name="x")

    assert prettyprinter.pformat(x) == "x"


def test_prettyprinter_formats_dunder_operators_with_precedence():
    prettyprinter = __import__("prettyprinter")
    x = defop(int, name="x")
    y = defop(int, name="y")

    assert prettyprinter.pformat(x() + y() * 2) == "x() + y() * 2"
    assert prettyprinter.pformat((x() + y()) * 2) == "(x() + y()) * 2"
    assert prettyprinter.pformat(x() - (y() - 2)) == "x() - (y() - 2)"
    assert prettyprinter.pformat((x() ** y()) ** 2) == "(x() ** y()) ** 2"
    assert prettyprinter.pformat(x() ** (y() ** 2)) == "x() ** y() ** 2"
    assert prettyprinter.pformat(-(x() ** 2)) == "-x() ** 2"
    assert prettyprinter.pformat((-x()) ** 2) == "(-x()) ** 2"


def test_interpretation_isinstance():
    a = defop(int)
    b = defop(str)

    assert isinstance({a: lambda: 0, b: lambda: "hello"}, Interpretation)
    assert not isinstance({a: 0, b: "hello"}, Interpretation)
    assert not isinstance([a, b], Interpretation)
    assert not isinstance({"a": lambda: 0, "b": lambda: "hello"}, Interpretation)


def test_term_str_hides_traversal_constructors():
    @dataclasses.dataclass
    class Box:
        value: object

    x = defop(int, name="x")

    @defop
    def outer(value: object) -> object:
        raise NotHandled

    term = outer({"box": Box(x())})

    assert str(term) == "outer({'box': Box(value=x())})"
    assert "_as_typ" not in str(term)
    assert "__apply__" not in str(term)


def test_instance_method_signature_excludes_self():
    """Instance-bound operations should not have 'self' in their signature.

    When an Operation is used as a method and accessed on an instance,
    __get__ creates a new Operation from a bound method. The signature
    should reflect the bound method (without 'self'), not the original
    unbound function.

    This failed with cached_property because functools.update_wrapper
    copied a stale __signature__ (with 'self') into __dict__, shadowing
    the descriptor.
    """

    class MyClass:
        @defop
        def my_method(self, x: int) -> str:
            raise NotHandled

    # Access the class-level signature first, which with cached_property
    # stores (self, x: int) -> str in MyClass.my_method.__dict__['__signature__'].
    # This is the key trigger: __get__ later copies __dict__ via functools.wraps
    # to the instance operation, shadowing a cached_property but not a property.
    cls_sig = MyClass.my_method.__signature__
    assert "self" in cls_sig.parameters  # class-level should have self

    instance = MyClass()
    instance_op = instance.my_method

    # The instance operation should have a signature without 'self'
    sig = inspect.signature(instance_op)
    assert "self" not in sig.parameters
    assert "x" in sig.parameters

    # Binding should work with just the real args (no 'self')
    sig.bind(42)


def test_defop_generic_typeddict_type_inference():
    """defop with generic TypedDict params should infer return type from nested dicts."""

    class Datum[T](typing.TypedDict):
        name: str
        value: T

    class Outer[T](typing.TypedDict):
        inner: Datum[T]

    @defop
    def unwrap_outer[T](x: Outer[T]) -> T:
        raise NotHandled

    term = unwrap_outer({"inner": {"name": "a", "value": 1}})

    assert typeof(term) == int


def test_bound_operation_names_what_it_is_bound_to():
    """A bound operation carries `__func__` and `__self__`, as a bound method does.

    `defdata` reads them to recognize a node it is rebuilding when the bound
    operation re-enters it under the class operation.
    """

    class MyClass:
        @defop
        def my_method(self, x: int) -> str:
            raise NotHandled

    instance = MyClass()
    bound = instance.my_method

    assert bound.__func__ is MyClass.my_method
    assert bound.__self__ is instance
    assert not hasattr(MyClass.my_method, "__self__")

    # The same operation bound to another instance is a distinct operation.
    other = MyClass()
    assert other.my_method is not bound
    assert other.my_method.__self__ is other
