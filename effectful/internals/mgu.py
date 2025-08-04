from typing import Any

from effectful.ops.syntax import deffn, syntactic_eq
from effectful.ops.types import Term


def mgu(*pairs, intp=None) -> dict | None:
    if intp is None:
        intp = {}
    for t1, t2 in pairs:
        intp = _mgu(t1, t2, intp)
    return intp


def _mgu(term1: Any, term2: Any, intp: dict) -> dict | None:
    """
    Unifier of term1 with term2.
    (Unlike usual mgu, this does not unify bi-directionally.)
    """
    # Case 1: terms are identical -> pass
    if syntactic_eq(term1, term2):
        return intp

    # Case 2: term1 is ground -> fail
    if not isinstance(term1, Term):
        return None

    # Case 3: term1 is an open var -> assign term2 to it
    if len(term1.args) == 0:
        if term1.op in intp:
            if syntactic_eq(term2, intp[term1.op]()):
                return intp
            else:
                return None
        else:
            intp[term1.op] = deffn(term2)
            return intp

    # Case 4: term1 is structured term -> check inductively
    if (
        isinstance(term2, Term)
        and term1.op == term2.op
        and len(term1.args) == len(term2.args)
    ):
        for t1, t2 in zip(term1.args, term2.args, strict=False):
            intp = _mgu(t1, t2, intp)  # type: ignore
            if intp is None:
                return None
        return intp
    return None
