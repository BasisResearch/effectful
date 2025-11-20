import numbers

import islpy as isl
from effectful.handlers.jax import jax_getitem
from effectful.handlers.jax import numpy as jnp
from effectful.ops.semantics import evaluate, fvsof, fwd, handler
from effectful.ops.syntax import ObjectInterpretation, deffn, defop, implements
from effectful.ops.types import Term

from weighted.ops.reduce import reduce


def _intp_add(lhs, rhs):
    if isinstance(lhs, numbers.Real):
        lhs = {1: lhs}
    if isinstance(rhs, numbers.Real):
        rhs = {1: rhs}
    if lhs.keys().isdisjoint(rhs.keys()):
        return lhs | rhs
    assert False


def _intp_arange(lower, upper):
    lower = {k: -v for k, v in lower.items()}
    upper[1] -= 1
    return lower, upper


def _intp_mul(lhs, rhs):
    if isinstance(rhs, numbers.Real):
        lhs, rhs = lhs, rhs

    if isinstance(lhs, numbers.Real):
        if isinstance(rhs, numbers.Real):
            return lhs * rhs
        elif isinstance(rhs, dict):
            return {k: lhs * v for k, v in rhs.items()}
    assert False


def arange_constraint(space, k, lower, upper):
    lower_constraint = isl.Constraint.ineq_from_names(space, lower | {str(k): 1})
    upper_constraint = isl.Constraint.ineq_from_names(space, upper | {str(k): -1})
    return [lower_constraint, upper_constraint]


def array_constraint(space, k, array):
    lower = {1: -jnp.min(array)}
    upper = {1: jnp.max(array)}
    return arange_constraint(space, k, lower, upper)


class ReduceLinearIndexer(ObjectInterpretation):
    """Tabular indexer for affine dependent streams."""

    @implements(reduce)
    def reduce(self, monoid, streams, body):
        stream_vars = set(streams.keys())
        linear_streams = {
            k: v for k, v in streams.items() if isinstance(v, Term) and v.op is jnp.arange
        }
        if len(linear_streams) == 0:
            return fwd()

        used_streams = (
            set.union(*(fvsof(v) for v in linear_streams.values())) & stream_vars
        )
        if any(fvsof(streams[k] for k in used_streams)):
            return fwd()

        op_names = tuple(used_streams | set(linear_streams.keys()))
        str_names = tuple(map(str, op_names))
        isl_space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, set=str_names)

        isl_intp = {k: deffn({str(k): 1}) for k in op_names}
        isl_intp |= {
            jnp.arange: _intp_arange,
            jnp.add: _intp_add,
            jnp.multiply: _intp_mul,
        }

        constraints = [array_constraint(isl_space, k, streams[k]) for k in used_streams]

        with handler(isl_intp):
            constraints += [
                arange_constraint(isl_space, k, *evaluate(v))
                for k, v in linear_streams.items()
            ]

        isl_set = isl.BasicSet.universe(isl_space)
        for constraint in sum(constraints, []):
            isl_set = isl_set.add_constraint(constraint)

        def add_point(point):
            val = point.get_multi_val()
            point = [val.get_at(i).to_python() for i in range(len(val))]
            points.append(point)

        points = []
        isl_set.to_set().foreach_point(add_point)
        points = jnp.array(points)

        new_index = defop(type(points))
        new_streams = {k: v for k, v in streams.items() if k not in op_names}
        new_streams[new_index] = jnp.arange(len(points))
        op_range = {op: points[:, op_names.index(op)] for op in op_names}
        linear_intp = {
            op: deffn(jax_getitem(op_range[op], (new_index(),))) for op in op_names
        }
        new_body = evaluate(body, intp=linear_intp)
        return reduce(monoid, new_streams, new_body)
