import json
import os.path
import time
import timeit
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng_factory():
    def _supply_rng():
        return np.random.default_rng(727)

    return _supply_rng


@pytest.fixture()
def rng(rng_factory):
    return rng_factory()


@pytest.fixture(scope="session")
def benchmark():
    bench_output_directory = os.path.join(os.path.dirname(__file__), "benchmark-report")
    # create utc timestamp formatted as "YYYY-mm-ddTHH-MM-SS+ZZ"
    utc_ts = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace(":", "")
        .replace("-", "")
    )
    bench_output_filename = f"output-{utc_ts}.jsonl"
    bench_output_file = os.path.join(bench_output_directory, bench_output_filename)

    def __benchmark_fn(
        func: Callable,
        calls: int = 1,
        iterations: int = 10,
        extra: Optional[dict[str, object]] = None,
    ):
        result = timeit.repeat(
            func, number=calls, repeat=iterations, timer=time.perf_counter_ns
        )

        os.makedirs(bench_output_directory, exist_ok=True)

        if extra is None:
            extra = {}

        with open(bench_output_file, "a", encoding="utf-8") as f:
            d = {"iterations": iterations, "calls": calls, "timings": result, **extra}

            f.write(json.dumps(d) + os.linesep)

    return __benchmark_fn
