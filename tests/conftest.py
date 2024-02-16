import json
import os.path
import time
import timeit
from datetime import datetime
from typing import Callable, Optional, Any

import numpy as np
import pytest


@pytest.fixture()
def rng():
    return np.random.default_rng(727)


@pytest.fixture(scope="session")
def benchmark():
    bench_output_directory = os.path.join(os.path.dirname(__file__), "benchmark-report")
    bench_output_filename = f"output-{int(datetime.now().timestamp())}.jsonl"
    bench_output_file = os.path.join(bench_output_directory, bench_output_filename)

    def __write_to_benchmark_file(
        func: Callable,
        result: list[float],
        calls: int,
        iterations: int,
        name: str | None = None,
        extra: Optional[dict[str, Any]] = None,
    ):
        os.makedirs(bench_output_directory, exist_ok=True)

        with open(bench_output_file, "a", encoding="utf-8") as f:
            d = {
                "name": func.__name__ if name is None else name,
                "iterations": iterations,
                "calls": calls,
                "timings": result,
            }

            if extra is not None and isinstance(extra, dict):
                d["extra"] = extra

            f.write(json.dumps(d) + os.linesep)

    def __benchmark_fn(
        func: Callable,
        calls: int = 1000,
        iterations: int = 5,
        name: str | None = None,
        extra: Optional[dict[str, Any]] = None,
    ):
        result = timeit.repeat(
            func, number=calls, repeat=iterations, timer=time.perf_counter_ns
        )
        __write_to_benchmark_file(func, result, calls, iterations, name, extra)

    return __benchmark_fn
