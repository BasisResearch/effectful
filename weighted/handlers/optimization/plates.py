from functools import cache, reduce
from itertools import product

import effectful.handlers.jax as handler
import jax
import numpy as np
from effectful.ops.semantics import coproduct, evaluate, fvsof, fwd
from effectful.ops.syntax import ObjectInterpretation, deffn, defop, implements
from effectful.ops.types import Term

import weighted.ops.fold as ops
from weighted.handlers.optimization.utils import parse_terms


@defop
def plated(streams, body):
    raise NotImplementedError


class LiftPlated(ObjectInterpretation):
    """
    Tensorized domain-lifted plating, c.f. [1, 2].

    As per the seminal dichotomy result of lifted inference [3, 4, 5],
    this algorithm can solve any polytime domain-liftable fold,
    and will forward iff inference is #P-hard.

    [1]: Taghipour, Nima, et al. "Lifted variable elimination:
      Decoupling the operators from the constraint language." JAIR. 2013.
    [2] Obermeyer, Fritz, et al. "Tensor variable elimination
      for plated factor graphs." ICML. 2019.
    [3] Dalvi, Nilesh, and Dan Suciu. "The dichotomy of probabilistic
      inference for unions of conjunctive queries." JACM. 2013.
    [4] Gribkoff, Eric, Guy Van den Broeck, and Dan Suciu.
      "Understanding the Complexity of Lifted Inference and
      Asymmetric Weighted Model Counting." UAI. 2014.
    [5] Taghipour, Nima, et al. "Completeness results for
      lifted variable elimination." AISTATS. 2013.
    """

    @implements(plated)
    def plated(self, plate_streams, body):
        # todo: implement
        return fwd()


@cache
def _substitute_fresh_plate_vars(ix, plate_vars):
    return defop(jax.Array, name=f"{ix}[{plate_vars}]")


def _unroll_vars(arr_ixs, plate_streams, fold_streams):
    new_fold_streams = {}
    new_arr_intp = {}
    for arr_ix in arr_ixs:
        match arr_ix:
            case Term(var_ix, ()):
                if var_ix in fold_streams:
                    new_fold_streams[var_ix] = fold_streams[var_ix]

            case Term(handler.jax_getitem, (Term(var_ix, ()), plates)):
                streams = (np.array(plate_streams[plate.op]) for plate in plates)
                streams = reduce(np.outer, streams)
                new_vars = np.zeros_like(streams, dtype=object)
                for ix, _ in np.ndenumerate(streams):
                    new_vars[ix] = _substitute_fresh_plate_vars(var_ix, ix)()
                    new_fold_streams[new_vars[ix].op] = fold_streams[var_ix]
                new_arr_intp[var_ix] = deffn(new_vars)
    return new_arr_intp, new_fold_streams


class PlateUnrolling(ObjectInterpretation):
    """
    Naive plate handling by unrolling.
    #P in the stream sizes of the plate indices.
    """

    @implements(plated)
    def plated(self, plate_streams, body):
        plate_vars = set(plate_streams.keys())
        match body:
            case Term(ops.fold, (monoid, fold_streams, fold_body)):
                mul, terms = parse_terms(fold_body, monoid)
                unrolled_terms = []
                unrolled_streams = {}

                for term in terms:
                    match term:
                        case Term(handler.jax_getitem, (_, arr_ixs)):
                            plates = tuple(fvsof(arr_ixs) & plate_vars)
                            vars_intp, new_streams = _unroll_vars(
                                arr_ixs, plate_streams, fold_streams
                            )
                            # create a new term for each grounding
                            plate_vals_iter = product(*(plate_streams[v] for v in plates))
                            for plate_vals in plate_vals_iter:
                                plate_intp = dict(zip(plates, plate_vals, strict=False))
                                plate_intp = {k: deffn(v) for k, v in plate_intp.items()}
                                intp = coproduct(plate_intp, vars_intp)
                                unrolled_terms.append(evaluate(term, intp=intp))
                                unrolled_streams |= new_streams
                        case _:
                            return fwd()

                unrolled_body = reduce(mul, unrolled_terms)
                return ops.fold(monoid, unrolled_streams, unrolled_body)
            case _:
                return fwd()


interpretation = coproduct(PlateUnrolling(), LiftPlated())
