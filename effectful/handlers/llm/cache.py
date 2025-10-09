import dataclasses
import weakref
from typing import Any

from effectful.ops.llm import Template
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


class TemplateCache(ObjectInterpretation):
    """Caches prompt template instantiations."""

    @dataclasses.dataclass(frozen=True, eq=True)
    class _ArgsKwargs:
        args: tuple[Any, ...]
        kwargs: tuple[tuple[str, Any], ...]

    _cache: weakref.WeakKeyDictionary[Template, dict[_ArgsKwargs, Any]]

    def __init__(self):
        self._cache = weakref.WeakKeyDictionary()

    @implements(Template.__call__)
    def _call(self, template, *args, **kwargs):
        call_cache = self._cache[template] if template in self._cache else {}
        key = TemplateCache._ArgsKwargs(tuple(args), tuple(kwargs.items()))

        try:
            in_call_cache = key in call_cache
        except TypeError as e:
            if "unhashable type" in str(e):
                return fwd()
            raise e

        if in_call_cache:
            return call_cache[key]

        result = fwd()
        call_cache[key] = result
        self._cache[template] = call_cache
        return result
