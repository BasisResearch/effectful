from effectful.ops.syntax import ObjectInterpretation


class SynthesisError(Exception):
    """Raised when program synthesis fails."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class ProgramSynthesis(ObjectInterpretation):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
