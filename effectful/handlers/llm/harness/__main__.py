"""Entry point for ``python -m effectful.handlers.llm.harness``.

The launcher itself lives in the package's `__init__`; this module only exists
because `runpy` requires a `__main__` submodule to execute a package.
"""

from effectful.handlers.llm.harness import main

if __name__ == "__main__":
    main()
