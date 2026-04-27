import inspect
import typing

from effectful.ops.semantics import typeof
from effectful.ops.syntax import defop
from effectful.ops.types import Interpretation, NotHandled


def test_interpretation_isinstance():
    a = defop(int)
    b = defop(str)

    assert isinstance({a: lambda: 0, b: lambda: "hello"}, Interpretation)
    assert not isinstance({a: 0, b: "hello"}, Interpretation)
    assert not isinstance([a, b], Interpretation)
    assert not isinstance({"a": lambda: 0, "b": lambda: "hello"}, Interpretation)


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
