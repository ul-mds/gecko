import numpy as np
import pytest

from gecko import generator

pytestmark = pytest.mark.benchmark


def __generate_numbers(n: int, rng: np.random.Generator):
    return generator.from_uniform_distribution(0, 2**8 - 1, rng=rng)(n)


def test_generate_100_numbers(benchmark, rng):
    def __generate_100_numbers():
        return __generate_numbers(100, rng)

    benchmark(__generate_100_numbers)


def test_generate_1000_numbers(benchmark, rng):
    def __generate_1000_numbers():
        return __generate_numbers(1000, rng)

    benchmark(__generate_1000_numbers)


def test_generate_10000_numbers(benchmark, rng):
    def __generate_10000_numbers():
        return __generate_numbers(10000, rng)

    benchmark(__generate_10000_numbers)
