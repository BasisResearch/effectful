import pydantic

from effectful.ops.syntax import ObjectInterpretation


class SynthesizedFunction(pydantic.BaseModel):
    """Structured output for function synthesis.

    Pydantic model representing synthesized code with function name and module code.
    """

    module_code: str = pydantic.Field(
        ...,
        description="Complete Python module code (no imports needed)",
    )


class SynthesisError(Exception):
    """Raised when program synthesis fails."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class ProgramSynthesis(ObjectInterpretation):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
