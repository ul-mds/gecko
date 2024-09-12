import os
from types import ModuleType
from typing import Callable

import numpy as np
import pytest
from _pytest.legacypath import TempdirFactory
from git import Repo

from gecko import generator, mutator
from tests.helpers import get_asset_path

pytestmark = pytest.mark.benchmark
record_counts = (100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000)


def __extra(mod_type: ModuleType, func_type: Callable, records: int, **kwargs: object):
    return {
        "suite": "german",
        "module": mod_type.__name__,
        "function": func_type.__name__,
        "n_records": records,
        **kwargs,
    }


@pytest.fixture(scope="module")
def gecko_data_path(tmpdir_factory: TempdirFactory):
    git_root_dir_path = tmpdir_factory.mktemp("git")
    repo = Repo.clone_from(
        "https://github.com/ul-mds/gecko-data.git",
        git_root_dir_path,
        no_checkout=True,
    )
    gecko_data_sha = os.environ.get(
        "PYTEST__GECKO_DATA_COMMIT_SHA", "9b7a073caa4fedbc6917152454039d9e005a799c"
    )
    repo.git.checkout(gecko_data_sha)

    yield git_root_dir_path


def __create_given_name_gender_generator(
    rng: np.random.Generator, gecko_data_base_path
):
    return generator.from_multicolumn_frequency_table(
        gecko_data_base_path / "de_DE" / "given-name-gender.csv",
        value_columns=["given_name", "gender"],
        freq_column="count",
        rng=rng,
    )


def __create_street_municip_postcode_generator(
    rng: np.random.Generator, gecko_data_base_path
):
    return generator.from_multicolumn_frequency_table(
        gecko_data_base_path / "de_DE" / "street-municipality-postcode.csv",
        value_columns=["street_name", "municipality", "postcode"],
        freq_column="count",
        rng=rng,
    )


def __create_last_name_generator(rng: np.random.Generator, gecko_data_base_path):
    return generator.from_frequency_table(
        gecko_data_base_path / "de_DE" / "last-name.csv",
        value_column="last_name",
        freq_column="count",
        rng=rng,
    )


def test_bench_single_column_generator(benchmark, rng, gecko_data_path):
    gen = __create_last_name_generator(rng, gecko_data_path)

    def __gen_single_column_last_name(n: int):
        return lambda: gen(n)

    for count in record_counts:
        benchmark(
            __gen_single_column_last_name(count),
            extra=__extra(
                generator,
                generator.from_frequency_table,
                count,
                csv_file="de_DE/last-name.csv",
                n_columns=1,
            ),
        )


def test_bench_double_column_generator(benchmark, rng, gecko_data_path):
    gen = __create_given_name_gender_generator(rng, gecko_data_path)

    def __gen_multi_column_given_name_gender(n: int):
        return lambda: gen(n)

    for count in record_counts:
        benchmark(
            __gen_multi_column_given_name_gender(count),
            extra=__extra(
                generator,
                generator.from_multicolumn_frequency_table,
                count,
                csv_file="de_DE/given-name-gender.csv",
                n_columns=2,
            ),
        )


def test_bench_triple_column_generator(benchmark, rng, gecko_data_path):
    gen = __create_street_municip_postcode_generator(rng, gecko_data_path)

    def __gen_multi_column_address(n: int):
        return lambda: gen(n)

    for count in record_counts:
        benchmark(
            __gen_multi_column_address(count),
            extra=__extra(
                generator,
                generator.from_multicolumn_frequency_table,
                count,
                csv_file="de_DE/street-municipality-postcode.csv",
                n_columns=3,
            ),
        )


def test_bench_to_data_frame(benchmark, rng, gecko_data_path):
    gen_single_col = __create_last_name_generator(rng, gecko_data_path)
    gen_multi_col_2 = __create_given_name_gender_generator(rng, gecko_data_path)
    gen_multi_col_3 = __create_street_municip_postcode_generator(rng, gecko_data_path)

    def __gen_dataframe(n: int):
        return lambda: generator.to_data_frame(
            {
                "last_name": gen_single_col,
                ("given_name", "gender"): gen_multi_col_2,
                ("street_name", "municipality", "postcode"): gen_multi_col_3,
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


def test_bench_mutate_edit(benchmark, rng, gecko_data_path):
    gen_last_name = __create_last_name_generator(rng, gecko_data_path)
    mut_edit = mutator.with_edit(rng=rng)

    for count in record_counts:
        srs_lst = gen_last_name(count)
        benchmark(
            lambda: mut_edit(srs_lst),
            extra=__extra(
                mutator,
                mutator.with_edit,
                count,
            ),
        )


def test_bench_corrupt_transpose(benchmark, rng, gecko_data_path):
    gen_last_name = __create_last_name_generator(rng, gecko_data_path)
    mut_transpose = mutator.with_transpose(rng=rng)

    for count in record_counts:
        srs_lst = gen_last_name(count)
        benchmark(
            lambda: mut_transpose(srs_lst),
            extra=__extra(
                mutator,
                mutator.with_transpose,
                count,
            ),
        )


def test_bench_corrupt_insert(benchmark, rng, gecko_data_path):
    gen_last_name = __create_last_name_generator(rng, gecko_data_path)
    mut_insert = mutator.with_insert(rng=rng)

    for count in record_counts:
        srs_lst = gen_last_name(count)
        benchmark(
            lambda: mut_insert(srs_lst),
            extra=__extra(
                mutator,
                mutator.with_insert,
                count,
            ),
        )


def test_bench_corrupt_delete(benchmark, rng, gecko_data_path):
    gen_last_name = __create_last_name_generator(rng, gecko_data_path)
    mut_delete = mutator.with_delete(rng=rng)

    for count in record_counts:
        srs_lst = gen_last_name(count)
        benchmark(
            lambda: mut_delete(srs_lst),
            extra=__extra(
                mutator,
                mutator.with_delete,
                count,
            ),
        )


def test_bench_corrupt_substitute(benchmark, rng, gecko_data_path):
    gen_last_name = __create_last_name_generator(rng, gecko_data_path)
    mut_sub = mutator.with_substitute(rng=rng)

    for count in record_counts:
        srs_lst = gen_last_name(count)
        benchmark(
            lambda: mut_sub(srs_lst),
            extra=__extra(
                mutator,
                mutator.with_substitute,
                count,
            ),
        )


def test_bench_corrupt_cldr(benchmark, rng, gecko_data_path):
    gen_last_name = __create_last_name_generator(rng, gecko_data_path)
    mut_cldr = mutator.with_cldr_keymap_file(
        get_asset_path("de-t-k0-windows.xml"),
        rng=rng,
    )

    for count in record_counts:
        srs_lst = gen_last_name(count)
        benchmark(
            lambda: mut_cldr(srs_lst),
            extra=__extra(
                mutator,
                mutator.with_cldr_keymap_file,
                count,
            ),
        )


def test_bench_corrupt_replacement_table(benchmark, rng, gecko_data_path):
    gen_last_name = __create_last_name_generator(rng, gecko_data_path)
    mut_ocr = mutator.with_replacement_table(
        gecko_data_path / "common" / "ocr.csv",
        inline=True,
        rng=rng,
    )

    for count in record_counts:
        srs_lst = gen_last_name(count)
        benchmark(
            lambda: mut_ocr(srs_lst),
            extra=__extra(
                mutator,
                mutator.with_replacement_table,
                count,
            ),
        )


def test_bench_corrupt_categorical(benchmark, rng, gecko_data_path):
    gen_given_name_gender = __create_given_name_gender_generator(rng, gecko_data_path)
    mut_categorical = mutator.with_categorical_values(
        gecko_data_path / "de_DE" / "given-name-gender.csv",
        value_column="gender",
        rng=rng,
    )

    for count in record_counts:
        srs_lst = gen_given_name_gender(count)
        benchmark(
            lambda: mut_categorical(srs_lst[1:]),
            extra=__extra(
                mutator,
                mutator.with_categorical_values,
                count,
            ),
        )


def test_bench_permute(benchmark, rng, gecko_data_path):
    gen_given_name_gender = __create_given_name_gender_generator(rng, gecko_data_path)
    mut_permute = mutator.with_permute()

    for count in record_counts:
        srs_lst = gen_given_name_gender(count)
        benchmark(
            lambda: mut_permute(srs_lst),
            extra=__extra(
                mutator,
                mutator.with_permute,
                count,
            ),
        )
