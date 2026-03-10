from RestrictedPython import PrintCollector, RestrictingNodeTransformer

from effectful.handlers.llm import Agent, Template, Tool
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.handlers.llm.evaluation import RestrictedEvalProvider, compile, parse
from effectful.handlers.llm.evaluation import (
    exec as effectful_exec,
)
from effectful.ops.semantics import handler


class DenyImports(RestrictingNodeTransformer):
    """Policy that forbids all import statements but otherwise inherits
    the default RestrictedPython restrictions (including print support)."""

    def visit_Import(self, node):
        self.error(node, "Imports are not allowed.")
        return node

    def visit_ImportFrom(self, node):
        self.error(node, "Imports are not allowed.")
        return node


class CodeAdaptAgent(Agent):
    """You are a genius problem solver and an expert Python programmer. You solve problems using a metacognitive approach: you think through challenging tasks using a blend of natural language reasoning and executable code - your natural language articulates both direct reasoning and strategic planning (meta-reasoning), while your code is interpreted and executed by a Python environment, allowing you to perform reasoning through computational operations. You excel at this way of writing reasoning programs.

    ## How to interact

    ### You ("Assistant")

    1. Think and plan in natural language

    2. Execute Python code using the `self__execute_python_code` tool. Anything that
    your code prints out will be returned to you.

    3. Return your final answer in your last message.

    ## Programming Environment

    You can use any Python builtins. The following libraries are preloaded and can be used directly:
    <code name="libraries">
    ```python
    import collections
    import copy
    from enum import Enum
    import itertools
    import json
    import math
    import random
    import re
    import string
    from typing import *
    ```

    You are NOT allowed to import or use any other libraries (trying to import
    or use other libraries will result in an error). These here are ALREADY
    IMPORTED, no need to import them.

    Variables persist between calls to `self__execute_python_code`.

    You do not have access to Internet links. Do not write asynchronous functions.

    ## Reasoning tips
    Here is a list of advice and information about how to reason well:
    - First analyze the problem. You can think about different possible solving strategies, evaluate them, then
    pick the most promising
    - Given that strategy, list all possible things that could go wrong, and find a way to prevent these errors
    and mistakes
    - Break problems into steps and subproblems whenever possible
    - Be obsessive about evaluating your answers and intermediate results
    - Verify that your solution meets all requirements, using code when possible
    - Code-based verification functions must provide useful feedback so you know what went wrong and how to
    improve your solution
    - Keep your code modular. Efficiently define and store important variables for later reuse
    - Use print() to inspect useful variables
    - Always write code---if there is any way to check your result using code, you should do so.
    """

    def __init__(self):
        self._globals = {}
        exec(
            "import collections\n"
            "import copy\n"
            "from enum import Enum\n"
            "import itertools\n"
            "import json\n"
            "import math\n"
            "import random\n"
            "import re\n"
            "import string\n"
            "from typing import *\n",
            self._globals,
        )
        self._globals["_print_"] = PrintCollector

    @Tool.define
    def execute_python_code(self, code: str) -> str:
        """Execute Python code in a persistent environment. Returns anything printed by the code."""
        print(f"\n{'=' * 60}")
        print("CODE:")
        print(f"{'=' * 60}")
        print(code)
        print(f"{'=' * 60}")

        filename = "<none>"

        module = parse(code, filename)
        compiled = compile(module, filename)

        effectful_exec(compiled, self._globals)

        # The RestrictedPython transformer rewrites print() calls to write
        # into a PrintCollector instance stored as _print in the globals.
        collector = self._globals.get("_print")
        output = collector() if callable(collector) else ""

        print("OUTPUT:")
        print(f"{'=' * 60}")
        print(output if output else "(no output)")
        print(f"{'=' * 60}\n")

        return output

    @Template.define
    def solve(self, problem: str) -> str:
        """{problem}"""
        pass


PROBLEMS = [
    r"""
    Let \(b\ge 2\) be an integer. Call a positive integer \(n\) \(b\text-\textit{eautiful}\) if it has exactly
two digits when expressed in base \(b\) and these two digits sum to \(\sqrt n\). For example, \(81\) is
\(13\text-\textit{eautiful}\) because \(81 = \underline{6} \ \underline{3}_{13} \) and \(6 + 3 = \sqrt{81}\).
Find the least integer \(b\ge 2\) for which there are more than ten \(b\text-\textit{eautiful}\) integers.
The final answer should be an integer between 0 and 999 (inclusive) with no additional formatting (only the
integer).
    """,
    """
    There are 3 people standing in a line. From left to right, they are numbered 1 to 3.
Each person has a set of attributes: Hobby, Movie-Genre, Sport.
The attributes have the following possible values:
Hobby: gardening, rock-climbing, singing
Movie-Genre: drama, fantasy, comedy
Sport: baseball, cricket, water-polo
Each person has a unique value for each attribute.
You know the following about the people:
The person who watches comedy and the person who plays water-polo have different parity positions
The person who likes rock-climbing is somewhere between the person who watches comedy and the person who
likes singing
The person who watches fantasy is not anywhere to the left of the person who watches comedy
The person who plays baseball is not anywhere to the right of the person who watches fantasy
The person who plays baseball and the person who watches fantasy have different parity positions
In the above, 'parity position' means the evenness or oddness of the person's position in the line.
Given this information, answer the following questions:
What is the movie genre of the person who plays cricket?
At what position is the person who watches drama?
At what position is the person who watches fantasy?
What sport does the person who likes singing play?
Think step by step and explain your reasoning, then output your answers in order in the format:
<solution>answer1, answer2, answer3, ...</solution>
For instance, if there were 3 questions and the answers were A, B, and C, the output would be:
<solution>A, B, C</solution>
If the answer to a question is a number, be sure to put it in numerical form (e.g. '3' instead of 'three').
    """,
    "Please generate a paragraph with exactly 4 sentences ending with 'walk', 'tumbling', 'another', and 'lunatic'.",
]

if __name__ == "__main__":
    provider = LiteLLMProvider(model="gpt-5-mini")

    with (
        handler(RestrictedEvalProvider(policy=DenyImports)),
        handler(provider),
        handler(RetryLLMHandler()),
    ):
        for problem_num, problem_statement in enumerate(PROBLEMS):
            agent = CodeAdaptAgent()
            print(f"Problem {problem_num}:")
            print("-" * 80)
            print("Statement:")
            print(problem_statement)
            print("-" * 80)
            answer = agent.solve(problem_statement)
            print("-" * 80)
            print("Answer:")
            print(answer)
            print()
