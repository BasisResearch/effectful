"""LLM-implemented functions via algebraic effects.

`effectful.handlers.llm` lets you write Python functions whose bodies are
implemented by a large language model, and call them like ordinary code.

## Core concepts

- **`Template`** — a fully type-annotated Python function whose body is `raise
  NotHandled` and whose docstring is a [format
  string](https://docs.python.org/3/library/string.html#format-string-syntax)
  prompt. Calling a template (under a provider) formats its arguments into the
  prompt, invokes the model, and decodes the response to the template's declared
  return type. Define one with the `Template.define` decorator.

- **`Tool`** — a normal Python callable exposed to the model. Its signature and
  docstring become the schema the model sees; the model calls it by name with
  JSON arguments and receives the encoded result. Tools in a template's lexical
  scope are offered to the model automatically; because scope is ordinary Python
  scope, an `Agent` (or an enclosing function) naturally partitions tools and
  templates into disjoint sets. Define one with `Tool.define`.

- **`Agent`** — a class mixin giving each instance a persistent message history,
  so its `Template` methods accumulate conversation context across calls.
  Instance attributes are available in prompts via `{self.attr}`.

- **`Encodable`** — the type-driven JSON bridge used internally to encode Python
  values into the model's context and decode the model's output (structured
  return values and tool-call arguments) back into typed Python objects.

## Tool calling and structured output

During a template call the model may take multiple turns: on each turn it can
call any `Tool` in scope (results are fed back and the loop continues) or
produce a final answer. The final answer is decoded to the template's return
type via constrained/structured generation, so non-`str` return types (ints,
dataclasses, etc.) come back as real Python values. A `FinalTool` lets the model
"answer" by calling a tool whose return value becomes the result and terminates
the loop.

## Providers and handlers

Execution is controlled by composing handlers with
`effectful.ops.semantics.handler(...)`: a provider such as
`effectful.handlers.llm.completions.LiteLLMProvider` implements the model calls,
and helpers like `RetryLLMHandler` add reliability behavior. Because everything
is an algebraic effect, behavior (model requests, tool dispatch, history) can be
observed, logged, or overridden by installing additional handlers.
"""

from .template import Agent, Template, Tool

__all__ = ["Agent", "Template", "Tool"]
