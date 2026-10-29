"""Versioned array-isolation and bounded-memory adapter for the frozen search."""
from contextlib import contextmanager

import numpy as np

from . import overnight_risk as original

_MAKE = original.make_representation
_KERNEL = original.representation_kernel
_RUN = original.run_job
BLOCK_ROWS = 1024


def make_representation(arrays, record, context, method, learner, arm, strength, geometry):
    isolated = dict(context)
    for key in ("angles", "metric"):
        if context[key] is not None:
            isolated[key] = {part: value.copy() for part, value in context[key].items()}
    return _MAKE(arrays, record, isolated, method, learner, arm, strength, geometry)


def representation_kernel(rep, left, right, gpu_device=None):
    part, selection = left
    positions = np.atleast_1d(np.arange(len(rep["values"][part]))[selection])
    right_part, right_selection = right
    columns = len(np.atleast_1d(np.arange(len(rep["values"][right_part]))[right_selection]))
    result = np.empty((len(positions), columns), dtype=np.float64)
    # Composite kernels need several pairwise temporaries in addition to the Gram.
    for start in range(0, len(positions), BLOCK_ROWS):
        block = positions[start:start + BLOCK_ROWS]
        result[start:start + len(block)] = _KERNEL(rep, (part, block), right, gpu_device)
    return result


@contextmanager
def implementation():
    previous_make, previous_kernel = original.make_representation, original.representation_kernel
    original.make_representation, original.representation_kernel = make_representation, representation_kernel
    try:
        yield
    finally:
        original.make_representation, original.representation_kernel = previous_make, previous_kernel


def run_job(job, stage, protocol, output, hashes):
    with implementation():
        return _RUN(job, stage, protocol, output, hashes)
