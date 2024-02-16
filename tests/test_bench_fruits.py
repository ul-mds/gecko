import numpy as np
import pytest

from gecko import generator, corruptor
from tests.helpers import get_asset_path

pytestmark = pytest.mark.benchmark
record_counts = (100, 500, 1000)


def __extra(fn_name: str, n: int):
    return {
        "suite": "fruits",
        "function_name": fn_name,
        "n_records": n,
    }


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
            __gen_multicolumn_fruits(count),
            name=f"gen_multicolumn_fruits({count})",
            extra=__extra("generator.from_multicolumn_frequency_table", count),
        )


def test_bench_single_column_generator(benchmark, rng):
    gen = __create_origin_single_column_generator(rng)

    def __gen_single_column_origin(n: int):
        return lambda: gen(n)

    for count in record_counts:
        benchmark(
            __gen_single_column_origin(count),
            name=f"gen_single_column_fruits({count})",
            extra=__extra("generator.from_frequency_table", count),
        )


def test_bench_normal_distribution(benchmark, rng):
    gen = __create_weight_normal_dist_generator(rng)

    def __gen_normal_dist_fruits(n: int):
        return lambda: gen(n)

    for count in record_counts:
        benchmark(
            __gen_normal_dist_fruits(count),
            name=f"gen_normal_dist_fruits({count})",
            extra=__extra("generator.from_normal_distribution", count),
        )


def test_bench_uniform_distribution(benchmark, rng):
    gen = __create_amount_uniform_dist_generator(rng)

    def __gen_uniform_dist_fruits(n: int):
        return lambda: gen(n)

    for count in record_counts:
        benchmark(
            __gen_uniform_dist_fruits(count),
            name=f"gen_uniform_dist_fruits({count})",
            extra=__extra("generator.from_uniform_distribution", count),
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
        benchmark(
            __gen_dataframe(count),
            name=f"gen_dataframe_fruits({count})",
            extra=__extra("generator.to_dataframe", count),
        )


def test_bench_corrupt_edit(benchmark, rng):
    gen_origin = __create_origin_single_column_generator(rng)
    corr_edit = corruptor.with_edit(rng=rng)

    for count in record_counts:
        srs_lst = gen_origin(count)
        benchmark(
            lambda: corr_edit(srs_lst),
            name=f"corr_edit_fruits({count})",
            calls=100,
            extra=__extra("corruptor.with_edit", count),
        )


def test_bench_corrupt_missing(benchmark, rng):
    gen_origin = __create_origin_single_column_generator(rng)
    corr_missing = corruptor.with_missing_value(strategy="all")

    for count in record_counts:
        srs_lst = gen_origin(count)
        benchmark(
            lambda: corr_missing(srs_lst),
            name=f"corr_missing_fruits({count})",
            calls=100,
            extra=__extra("corruptor.with_missing_value", count),
        )


def test_bench_corrupt_replacement_table(benchmark, rng):
    gen_origin = __create_origin_single_column_generator(rng)
    corr_replacement = corruptor.with_replacement_table(
        get_asset_path("ocr.csv"),
        rng=rng,
    )

    for count in record_counts:
        srs_lst = gen_origin(count)
        benchmark(
            lambda: corr_replacement(srs_lst),
            name=f"corr_replacement_fruits({count})",
            calls=100,
            extra=__extra("corruptor.with_replacement_table", count),
        )


def test_bench_corrupt_cldr(benchmark, rng):
    gen_origin = __create_origin_single_column_generator(rng)
    corr_cldr = corruptor.with_cldr_keymap_file(
        get_asset_path("de-t-k0-windows.xml"),
        rng=rng,
    )

    for count in record_counts:
        srs_lst = gen_origin(count)
        benchmark(
            lambda: corr_cldr(srs_lst),
            name=f"corr_cldr_fruits({count})",
            calls=100,
            extra=__extra("corruptor.with_cldr_keymap_file", count),
        )


def test_bench_permute(benchmark, rng):
    gen_fruit_type = __create_fruit_type_multicolumn_generator(rng)
    corr_permute = corruptor.with_permute()

    for count in record_counts:
        srs_lst = gen_fruit_type(count)
        benchmark(
            lambda: corr_permute(srs_lst),
            name=f"corr_permute_fruits({count})",
            calls=100,
            extra=__extra("corruptor.with_permute", count),
        )


def test_bench_categorical(benchmark, rng):
    gen_origin = __create_origin_single_column_generator(rng)
    corr_categorical = corruptor.with_categorical_values(
        get_asset_path("freq-fruit-origin.csv"),
        header=True,
        value_column="origin",
        rng=rng,
    )

    for count in record_counts:
        srs_lst = gen_origin(count)
        benchmark(
            lambda: corr_categorical(srs_lst),
            name=f"corr_categorical_fruits({count})",
            calls=100,
            extra=__extra("corruptor.with_categorical_values", count),
        )
