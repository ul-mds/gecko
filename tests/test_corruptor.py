import string

import pandas as pd

from geco.corruptor import (
    with_insert,
    with_delete,
    with_transpose,
    with_missing_value,
    with_substitute,
    with_categorical_values,
    with_edit,
    with_cldr_keymap_file,
    with_phonetic_replacement_table,
    with_replacement_table,
)
from tests.helpers import get_asset_path


def test_with_value_replace_all():
    x = pd.Series(["foo", "   ", ""])
    corr = with_missing_value("bar", "all")
    assert (corr(x) == pd.Series(["bar", "bar", "bar"])).all()


def test_with_value_replace_empty():
    x = pd.Series(["foo", "   ", ""])
    corr = with_missing_value("bar", "empty")
    assert (corr(x) == pd.Series(["foo", "   ", "bar"])).all()


def test_with_value_replace_blank():
    x = pd.Series(["foo", "   ", ""])
    corr = with_missing_value("bar", "blank")
    assert (corr(x) == pd.Series(["foo", "bar", "bar"])).all()


def test_with_random_insert(rng):
    x = pd.Series(["foo", "bar", "baz"])
    corr = with_insert(charset="x", rng=rng)
    x_corr = corr(x)

    # check that series are of the same length
    assert len(x) == len(x_corr)
    # check that all strings are different from one another
    assert ~(x == x_corr).all()

    # check that all string pairs are different in only one char
    for i in range(len(x)):
        assert len(x.iloc[i]) + 1 == len(x_corr.iloc[i])
        # check that this char is the `x`
        assert "x" not in x.iloc[i]
        assert "x" in x_corr.iloc[i]


def test_with_random_delete(rng):
    x = pd.Series(["foo", "bar", "baz"])
    corr = with_delete(rng=rng)
    x_corr = corr(x)

    # check that series are of the same length
    assert len(x) == len(x_corr)
    # check that all strings are different from one another
    assert ~(x == x_corr).all()

    # check that all string pairs are different in one char
    assert ((x.str.len() - 1) == x_corr.str.len()).all()


def test_with_random_transpose(rng):
    x = pd.Series(["abc", "def", "ghi"])
    corr = with_transpose(rng=rng)
    x_corr = corr(x)

    # same lengths
    assert len(x) == len(x_corr)
    # all different
    assert ~(x == x_corr).all()
    # same string lengths
    assert (x.str.len() == x_corr.str.len()).all()

    # check that the characters are the same in both series
    for i in range(len(x)):
        assert set(x.iloc[i]) == set(x_corr.iloc[i])


def test_with_random_substitute(rng):
    x = pd.Series(["foo", "bar", "baz"])
    corr = with_substitute(charset="x", rng=rng)
    x_corr = corr(x)

    # same len
    assert len(x) == len(x_corr)
    # all different
    assert ~(x == x_corr).all()
    # same string lengths
    assert (x.str.len() == x_corr.str.len()).all()

    # check that original doesn't contain x
    assert (~x.str.contains("x")).all()
    # check that corrupted copy contains x
    assert x_corr.str.contains("x").all()


def test_with_categorical_values(rng):
    def _generate_gender_list():
        nonlocal rng
        return rng.choice(["m", "f", "d", "x"], size=1000)

    corr = with_categorical_values(
        get_asset_path("freq_table_gender.csv"), header=True, value_column="gender"
    )

    x = pd.Series(_generate_gender_list())
    x_corr = corr(x)

    # same length
    assert len(x) == len(x_corr)
    # different items
    assert ~(x == x_corr).all()


def test_with_edit(rng):
    def _new_string():
        nonlocal rng
        chars = list(string.ascii_letters)
        rng.shuffle(chars)
        return "".join(chars[:10])

    def _generate_strings():
        nonlocal rng
        return [_new_string() for _ in range(1000)]

    x = pd.Series(_generate_strings())
    corr = with_edit(
        p_insert=0.25,
        p_delete=0.25,
        p_substitute=0.25,
        p_transpose=0.25,
        charset=string.ascii_letters,
        rng=rng,
    )
    x_corr = corr(x)

    assert len(x) == len(x_corr)
    assert ~(x == x_corr).all()


def test_with_phonetic_replacement_table(rng):
    df_phonetic_in_out = pd.read_csv(get_asset_path("phonetic-test.csv"))
    srs_original = df_phonetic_in_out["original"]
    srs_corrupt = df_phonetic_in_out["corrupt"]

    corr = with_phonetic_replacement_table(get_asset_path("homophone-de.csv"), rng=rng)
    x_corr = corr(srs_original)

    assert (x_corr == srs_corrupt).all()


def test_with_replacement_table(rng):
    df_ocr_in_out = pd.read_csv(get_asset_path("ocr-test.csv"))
    srs_original = df_ocr_in_out["original"]
    srs_corrupt = df_ocr_in_out["corrupt"]

    corr = with_replacement_table(get_asset_path("ocr.csv"), rng=rng, p=1)
    x_corr = corr(srs_original)

    assert (x_corr == srs_corrupt).all()


def test_with_cldr_keymap_file(rng):
    x = pd.Series(["d", "e"])
    corr = with_cldr_keymap_file(get_asset_path("de-t-k0-windows.xml"))
    x_corr = corr(x)

    assert len(x) == len(x_corr)
    assert (x.str.len() == x_corr.str.len()).all()
    assert ~(x == x_corr).all()

    assert x_corr.iloc[0] in "Decsf"  # neighboring keys of `d`
    assert x_corr.iloc[1] in "E3dwr"  # neighboring keys of `e`


def test_with_cldr_keymap_file_no_replacement(rng):
    # this should stay the same since á is not mapped in the keymap
    x = pd.Series(["á"])
    corr = with_cldr_keymap_file(get_asset_path("de-t-k0-windows.xml"))
    x_corr = corr(x)

    assert len(x) == len(x_corr)
    assert (x.str.len() == x_corr.str.len()).all()
    assert (x == x_corr).all()
