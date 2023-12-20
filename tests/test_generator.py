import numpy as np
import pandas as pd

from geco import generator
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
