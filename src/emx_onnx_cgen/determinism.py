from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Iterator

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


@contextmanager
def deterministic_reference_runtime() -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in THREAD_ENV_VARS}
    for name in THREAD_ENV_VARS:
        os.environ[name] = "1"
    limits_context = None
    try:
        try:
            from threadpoolctl import threadpool_limits
        except Exception:
            threadpool_limits = None
        if threadpool_limits is not None:
            limits_context = threadpool_limits(limits=1)
            limits_context.__enter__()
        yield
    finally:
        if limits_context is not None:
            limits_context.__exit__(None, None, None)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
