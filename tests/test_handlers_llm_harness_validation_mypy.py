import gc
import os
import subprocess

import pytest

from effectful.handlers.llm.harness.validation.mypy import MypyTypeChecker


def test_mypy_reuses_mode_specific_caches_and_fresh_source_paths(monkeypatch):
    calls: list[list[str]] = []

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    checker = MypyTypeChecker()

    checker.type_check("x: int = 1\n")
    checker.type_check("x: int = 2\n")
    checker.type_check("x: int = 3\n", lenient=True)
    checker.type_check("x: int = 4\n", lenient=True)

    cache_dirs = [args[args.index("--cache-dir") + 1] for args in calls]
    assert cache_dirs[0] == cache_dirs[1]
    assert cache_dirs[2] == cache_dirs[3]
    assert cache_dirs[0] != cache_dirs[2]

    source_paths = [args[3] for args in calls]
    assert len(set(source_paths)) == len(source_paths)
    assert all(not os.path.exists(path) for path in source_paths)

    cache_root = checker._cache_root.name
    assert all(os.path.dirname(path) == cache_root for path in cache_dirs)
    assert os.path.isdir(cache_root)
    del checker
    gc.collect()
    assert not os.path.exists(cache_root)


def test_mypy_rechecks_changed_same_size_source_with_shared_cache():
    checker = MypyTypeChecker()
    checker.type_check("x: int = 1\n")

    # The replacement has the same byte length. A stable source path and mtime can
    # make mypy's fast cache check miss this change; type_check uses a fresh path.
    with pytest.raises(TypeError, match="Incompatible types in assignment"):
        checker.type_check("x: str = 1\n")
