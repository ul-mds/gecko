import numpy as np
import pandas as pd

from gecko import generator
from tests.helpers import get_asset_path


def test_from_function():
    flag = False

    def _generator() -> str:
        nonlocal flag
        flag = not flag

        return "foo" if flag else "bar"

    generate_foobar = generator.from_function(_generator)
    foobar_list = generate_foobar(4)

    assert len(foobar_list) == 1
    assert foobar_list[0].equals(pd.Series(["foo", "bar", "foo", "bar"]))


def test_from_uniform_distribution(rng):
    generate_uniform = generator.from_uniform_distribution(1, 10, rng=rng)
    number_list = generate_uniform(1000)

    assert len(number_list) == 1
    assert len(number_list[0]) == 1000

    numbers_as_floats = np.array([float(n) for n in number_list[0]], dtype=float)

    assert (numbers_as_floats >= 1).all()
    assert (numbers_as_floats <= 10).all()


def test_from_normal_distribution(rng):
    generate_normal = generator.from_normal_distribution(0, 1, rng=rng)
    number_list = generate_normal(1000)

    # bound checks don't really make sense here ...
    assert len(number_list) == 1
    assert len(number_list[0]) == 1000


def test_from_frequency_table_no_header(rng, foobar_freq_head):
    generate_tab = generator.from_frequency_table(
        get_asset_path("freq_table_no_header.csv"),
        rng=rng,
    )
    h = generate_tab(len(foobar_freq_head))[0]
    assert h.equals(pd.Series(foobar_freq_head))


def test_from_frequency_table_with_header(rng, foobar_freq_head):
    generate_tab = generator.from_frequency_table(
        get_asset_path("freq_table_header.csv"),
        rng=rng,
        header=True,
        value_column="value",
        freq_column="freq",
    )
    h = generate_tab(len(foobar_freq_head))[0]
    assert h.equals(pd.Series(foobar_freq_head))


def test_from_frequency_table_tsv(rng, foobar_freq_head):
    generate_tab = generator.from_frequency_table(
        get_asset_path("freq_table_no_header.tsv"), rng=rng, delimiter="\t"
    )
    h = generate_tab(len(foobar_freq_head))[0]
    assert h.equals(pd.Series(foobar_freq_head))


def test_from_frequency_table(rng):
    gen_fruits = generator.from_frequency_table(
        get_asset_path("freq-fruits.csv"),
        header=True,
        value_column="fruit",
        freq_column="count",
        rng=rng,
    )

    list_of_srs = gen_fruits(100)
    assert len(list_of_srs) == 1

    srs = list_of_srs[0]
    assert len(srs) == 100
    assert sorted(srs.unique()) == ["apple", "banana", "orange"]


def test_from_multicolumn_frequency_table(rng):
    gen_fruit_types = generator.from_multicolumn_frequency_table(
        get_asset_path("freq-fruits-types.csv"),
        header=True,
        value_columns=["fruit", "type"],
        freq_column="count",
        rng=rng,
    )

    list_of_srs = gen_fruit_types(100)
    assert len(list_of_srs) == 2

    srs_fruit, srs_type = list_of_srs
    assert len(srs_fruit) == 100
    assert len(srs_type) == 100

    for i in range(100):
        fruit = srs_fruit.iloc[i]
        fruit_type = srs_type.iloc[i]

        if fruit == "apple":
            assert fruit_type in ("braeburn", "elstar")
        elif fruit == "banana":
            assert fruit_type in ("cavendish", "plantain")
        elif fruit == "orange":
            assert fruit_type in ("clementine", "mandarin")
        else:
            raise AssertionError(f"unknown fruit: `{fruit}`")
