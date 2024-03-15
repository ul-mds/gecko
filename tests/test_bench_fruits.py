from types import ModuleType
from typing import Any, Callable

import numpy as np
import pytest

from gecko import generator, mutator
from tests.helpers import get_asset_path

pytestmark = pytest.mark.benchmark
record_counts = (100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000)


def __extra(
    mod_type: ModuleType, func_type: Callable, records: int, **kwargs: dict[str, Any]
):
    return {
        "suite": "fruits",
        "module": mod_type.__name__,
        "function": func_type.__name__,
        "n_records": records,
        **kwargs,
    }


def __create_fruit_type_multicolumn_generator(rng: np.random.Generator):
    return generator.from_multicolumn_frequency_table(
        get_asset_path("freq-fruits-types.csv"),
        value_columns=["fruit", "type"],
        freq_column="count",
        rng=rng,
    )


def __create_origin_single_column_generator(rng: np.random.Generator):
    return generator.from_frequency_table(
        get_asset_path("freq-fruit-origin.csv"),
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
            extra=__extra(
                generator,
                generator.from_multicolumn_frequency_table,
                count,
            ),
        )


def test_bench_single_column_generator(benchmark, rng):
    gen = __create_origin_single_column_generator(rng)

    def __gen_single_column_origin(n: int):
        return lambda: gen(n)

    for count in record_counts:
        benchmark(
            __gen_single_column_origin(count),
            extra=__extra(
                generator,
                generator.from_frequency_table,
                count,
            ),
        )


def test_bench_normal_distribution(benchmark, rng):
    gen = __create_weight_normal_dist_generator(rng)

    def __gen_normal_dist_fruits(n: int):
        return lambda: gen(n)

    for count in record_counts:
        benchmark(
            __gen_normal_dist_fruits(count),
            extra=__extra(
                generator,
                generator.from_normal_distribution,
                count,
            ),
        )


def test_bench_uniform_distribution(benchmark, rng):
    gen = __create_amount_uniform_dist_generator(rng)

    def __gen_uniform_dist_fruits(n: int):
        return lambda: gen(n)

    for count in record_counts:
        benchmark(
            __gen_uniform_dist_fruits(count),
            extra=__extra(
                generator,
                generator.from_uniform_distribution,
                count,
            ),
        )


def test_bench_to_dataframe(benchmark, rng):
    gen_single_col = __create_origin_single_column_generator(rng)
    gen_multi_col = __create_fruit_type_multicolumn_generator(rng)
    gen_normal = __create_weight_normal_dist_generator(rng)
    gen_uniform = __create_amount_uniform_dist_generator(rng)

    def __gen_dataframe(n: int):
        return lambda: generator.to_data_frame(
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
            extra=__extra(
                generator,
                generator.to_data_frame,
                count,
            ),
        )


def test_bench_mutate_edit(benchmark, rng):
    gen_origin = __create_origin_single_column_generator(rng)
    mut_edit = mutator.with_edit(rng=rng)

    for count in record_counts:
        srs_lst = gen_origin(count)
        benchmark(
            lambda: mut_edit(srs_lst),
            extra=__extra(
                mutator,
                mutator.with_edit,
                count,
            ),
        )


def test_bench_mutate_insert(benchmark, rng):
    gen_origin = __create_origin_single_column_generator(rng)
    mut_insert = mutator.with_insert(rng=rng)

    for count in record_counts:
        srs_lst = gen_origin(count)
        benchmark(
            lambda: mut_insert(srs_lst),
            extra=__extra(
                mutator,
                mutator.with_insert,
                count,
            ),
        )


def test_bench_mutate_delete(benchmark, rng):
    gen_origin = __create_origin_single_column_generator(rng)
    mut_delete = mutator.with_delete(rng=rng)

    for count in record_counts:
        srs_lst = gen_origin(count)
        benchmark(
            lambda: mut_delete(srs_lst),
            extra=__extra(
                mutator,
                mutator.with_delete,
                count,
            ),
        )


def test_bench_mutate_substitute(benchmark, rng):
    gen_origin = __create_origin_single_column_generator(rng)
    mut_substitute = mutator.with_substitute(rng=rng)

    for count in record_counts:
        srs_lst = gen_origin(count)
        benchmark(
            lambda: mut_substitute(srs_lst),
            extra=__extra(
                mutator,
                mutator.with_substitute,
                count,
            ),
        )


def test_bench_mutate_transpose(benchmark, rng):
    gen_origin = __create_origin_single_column_generator(rng)
    mut_transpose = mutator.with_transpose(rng=rng)

    for count in record_counts:
        srs_lst = gen_origin(count)
        benchmark(
            lambda: mut_transpose(srs_lst),
            extra=__extra(
                mutator,
                mutator.with_transpose,
                count,
            ),
        )


def test_bench_mutate_missing(benchmark, rng):
    gen_origin = __create_origin_single_column_generator(rng)
    mut_missing = mutator.with_missing_value(strategy="all")

    for count in record_counts:
        srs_lst = gen_origin(count)
        benchmark(
            lambda: mut_missing(srs_lst),
            extra=__extra(
                mutator,
                mutator.with_missing_value,
                count,
            ),
        )


def test_bench_mutate_replacement_table(benchmark, rng):
    gen_origin = __create_origin_single_column_generator(rng)
    mut_replacement = mutator.with_replacement_table(
        get_asset_path("ocr.csv"),
        rng=rng,
    )

    for count in record_counts:
        srs_lst = gen_origin(count)
        benchmark(
            lambda: mut_replacement(srs_lst),
            extra=__extra(
                mutator,
                mutator.with_replacement_table,
                count,
            ),
        )


def test_bench_mutate_cldr(benchmark, rng):
    gen_origin = __create_origin_single_column_generator(rng)
    mut_cldr = mutator.with_cldr_keymap_file(
        get_asset_path("de-t-k0-windows.xml"),
        rng=rng,
    )

    for count in record_counts:
        srs_lst = gen_origin(count)
        benchmark(
            lambda: mut_cldr(srs_lst),
            extra=__extra(
                mutator,
                mutator.with_cldr_keymap_file,
                count,
            ),
        )


def test_bench_permute(benchmark, rng):
    gen_fruit_type = __create_fruit_type_multicolumn_generator(rng)
    mut_permute = mutator.with_permute()

    for count in record_counts:
        srs_lst = gen_fruit_type(count)
        benchmark(
            lambda: mut_permute(srs_lst),
            extra=__extra(
                mutator,
                mutator.with_permute,
                count,
            ),
        )


def test_bench_categorical(benchmark, rng):
    gen_origin = __create_origin_single_column_generator(rng)
    mut_categorical = mutator.with_categorical_values(
        get_asset_path("freq-fruit-origin.csv"),
        value_column="origin",
        rng=rng,
    )

    for count in record_counts:
        srs_lst = gen_origin(count)
        benchmark(
            lambda: mut_categorical(srs_lst),
            extra=__extra(
                mutator,
                mutator.with_categorical_values,
                count,
            ),
        )
