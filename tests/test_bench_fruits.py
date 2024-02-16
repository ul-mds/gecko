import numpy as np
import pytest

from gecko import generator
from tests.helpers import get_asset_path

pytestmark = pytest.mark.benchmark
record_counts = (100, 500, 1000)


def __create_fruit_type_multicolumn_generator(rng: np.random.Generator):
    return generator.from_multicolumn_frequency_table(
        get_asset_path("freq-fruits-types.csv"),
        header=True,
        value_columns=["fruit", "type"],
        freq_column="count",
        rng=rng,
    )


def __create_origin_single_column_generator(rng: np.random.Generator):
    return generator.from_frequency_table(
        get_asset_path("freq-fruit-origin.csv"),
        header=True,
        value_column="origin",
        freq_column="count",
        rng=rng,
    )


def __create_weight_normal_dist_generator(rng: np.random.Generator):
    return generator.from_normal_distribution(
        mean=150,
        sd=50,
        precision=1,
        rng=rng,
    )


def __create_amount_uniform_dist_generator(rng: np.random.Generator):
    return generator.from_uniform_distribution(
        2,
        8,
        precision=0,
        rng=rng,
    )


def test_bench_multicolumn_generator(benchmark, rng):
    gen = __create_fruit_type_multicolumn_generator(rng)

    def __gen_multicolumn_fruits(n: int):
        return lambda: gen(n)

    for count in record_counts:
        benchmark(
            __gen_multicolumn_fruits(count), name=f"gen_multicolumn_fruits({count})"
        )


def test_bench_single_column_generator(benchmark, rng):
    gen = __create_origin_single_column_generator(rng)

    def __gen_single_column_origin(n: int):
        return lambda: gen(n)

    for count in record_counts:
        benchmark(
            __gen_single_column_origin(count), name=f"gen_single_column_fruits({count})"
        )


def test_bench_normal_distribution(benchmark, rng):
    gen = __create_weight_normal_dist_generator(rng)

    def __gen_normal_dist_fruits(n: int):
        return lambda: gen(n)

    for count in record_counts:
        benchmark(
            __gen_normal_dist_fruits(count), name=f"gen_normal_dist_fruits({count})"
        )


def test_bench_uniform_distribution(benchmark, rng):
    gen = __create_amount_uniform_dist_generator(rng)

    def __gen_uniform_dist_fruits(n: int):
        return lambda: gen(n)

    for count in record_counts:
        benchmark(
            __gen_uniform_dist_fruits(count), name=f"gen_uniform_dist_fruits({count})"
        )


def test_bench_to_dataframe(benchmark, rng):
    gen_single_col = __create_origin_single_column_generator(rng)
    gen_multi_col = __create_fruit_type_multicolumn_generator(rng)
    gen_normal = __create_weight_normal_dist_generator(rng)
    gen_uniform = __create_amount_uniform_dist_generator(rng)

    def __gen_dataframe(n: int):
        return lambda: generator.to_dataframe(
            {
                ("fruit", "type"): gen_multi_col,
                "origin": gen_single_col,
                "weight": gen_normal,
                "amount": gen_uniform,
            },
            n,
        )

    for count in record_counts:
        benchmark(__gen_dataframe(count), name=f"gen_dataframe_fruits({count})")
