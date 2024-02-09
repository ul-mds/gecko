import string

import pandas as pd
import pytest

from gecko.corruptor import (
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
    corrupt_dataframe,
    with_noop,
    with_function,
    with_permute,
)
from tests.helpers import get_asset_path


def test_with_function(rng):
    # basic corruptor that simply adds a random number from 0 to 9
    def _corruptor(value: str, rand) -> str:
        return value + str(rand.integers(0, 9))

    srs = pd.Series(["foo", "bar", "baz"])
    corrupt_ints = with_function(_corruptor, rand=rng)
    (srs_corrupted,) = corrupt_ints([srs])

    for i in range(len(srs)):
        x_orig, x_corr = srs.iloc[i], srs_corrupted.iloc[i]

        assert x_orig != x_corr
        assert len(x_corr) == len(x_orig) + 1
        assert x_corr[-1:] in string.digits


def test_with_value_replace_all():
    srs = pd.Series(["foo", "   ", ""])
    corrupt_missing = with_missing_value("bar", "all")
    (srs_corrupted,) = corrupt_missing([srs])

    assert (srs_corrupted == pd.Series(["bar", "bar", "bar"])).all()


def test_with_value_replace_empty():
    srs = pd.Series(["foo", "   ", ""])
    corrupt_missing = with_missing_value("bar", "empty")
    (srs_corrupted,) = corrupt_missing([srs])

    assert (srs_corrupted == pd.Series(["foo", "   ", "bar"])).all()


def test_with_value_replace_blank():
    srs = pd.Series(["foo", "   ", ""])
    corrupt_missing = with_missing_value("bar", "blank")
    (srs_corrupted,) = corrupt_missing([srs])

    assert (srs_corrupted == pd.Series(["foo", "bar", "bar"])).all()


def test_with_random_insert(rng):
    srs = pd.Series(["foo", "bar", "baz"])
    corrupt_insert = with_insert(charset="x", rng=rng)
    (srs_corrupted,) = corrupt_insert([srs])

    # check that series are of the same length
    assert len(srs) == len(srs_corrupted)
    # check that all strings are different from one another
    assert ~(srs == srs_corrupted).all()

    # check that all string pairs are different in only one char
    for i in range(len(srs)):
        assert len(srs.iloc[i]) + 1 == len(srs_corrupted.iloc[i])
        # check that this char is the `x`
        assert "x" not in srs.iloc[i]
        assert "x" in srs_corrupted.iloc[i]


def test_with_random_delete(rng):
    srs = pd.Series(["foo", "bar", "baz"])
    corrupt_delete = with_delete(rng=rng)
    (srs_corrupted,) = corrupt_delete([srs])

    # check that series are of the same length
    assert len(srs) == len(srs_corrupted)
    # check that all strings are different from one another
    assert ~(srs == srs_corrupted).all()
    # check that all string pairs are different in one char
    assert ((srs.str.len() - 1) == srs_corrupted.str.len()).all()


def test_with_random_delete_empty_string(rng):
    srs = pd.Series(["", "f"])
    corrupt_delete = with_delete(rng=rng)
    (srs_corrupted,) = corrupt_delete([srs])

    assert len(srs) == len(srs_corrupted)
    assert (srs_corrupted == "").all()


def test_with_random_transpose(rng):
    srs = pd.Series(["abc", "def", "ghi"])
    corrupt_transpose = with_transpose(rng=rng)
    (srs_corrupted,) = corrupt_transpose([srs])

    # same lengths
    assert len(srs) == len(srs_corrupted)
    # all different
    assert ~(srs == srs_corrupted).all()
    # same string lengths
    assert (srs.str.len() == srs_corrupted.str.len()).all()

    # check that the characters are the same in both series
    for i in range(len(srs)):
        assert set(srs.iloc[i]) == set(srs_corrupted.iloc[i])


def test_with_random_transpose_no_neighbor(rng):
    srs = pd.Series(["", "a", "ab"])
    corrupt_transpose = with_transpose(rng=rng)
    (srs_corrupted,) = corrupt_transpose([srs])

    # same lengths
    assert len(srs) == len(srs_corrupted)
    # none transposed except last
    assert (srs_corrupted == ["", "a", "ba"]).all()


def test_with_random_substitute(rng):
    srs = pd.Series(["foo", "bar", "baz"])
    corrupt_substitute = with_substitute(charset="x", rng=rng)
    (srs_corrupted,) = corrupt_substitute([srs])

    # same len
    assert len(srs) == len(srs_corrupted)
    # all different
    assert ~(srs == srs_corrupted).all()
    # same string lengths
    assert (srs.str.len() == srs_corrupted.str.len()).all()

    # check that original doesn't contain x
    assert (~srs.str.contains("x")).all()
    # check that corrupted copy contains x
    assert srs_corrupted.str.contains("x").all()


def test_with_random_substitute_empty_string(rng):
    srs = pd.Series(["", "f"])
    corrupt_substitute = with_substitute(charset="x", rng=rng)
    (srs_corrupted,) = corrupt_substitute([srs])

    # same len
    assert len(srs) == len(srs_corrupted)
    assert (srs_corrupted == ["", "x"]).all()


def test_with_categorical_values(rng):
    def _generate_gender_list():
        nonlocal rng
        return rng.choice(["m", "f", "d", "x"], size=1000)

    corrupt_categorical = with_categorical_values(
        get_asset_path("freq_table_gender.csv"), header=True, value_column="gender"
    )

    srs = pd.Series(_generate_gender_list())
    (srs_corrupted,) = corrupt_categorical([srs])

    # same length
    assert len(srs) == len(srs_corrupted)
    # different items
    assert ~(srs == srs_corrupted).all()


def test_with_edit(rng):
    def _new_string():
        nonlocal rng
        chars = list(string.ascii_letters)
        rng.shuffle(chars)
        return "".join(chars[:10])

    def _generate_strings():
        nonlocal rng
        return [_new_string() for _ in range(1000)]

    srs = pd.Series(_generate_strings())
    corrupt_edit = with_edit(
        p_insert=0.25,
        p_delete=0.25,
        p_substitute=0.25,
        p_transpose=0.25,
        charset=string.ascii_letters,
        rng=rng,
    )
    (srs_corrupted,) = corrupt_edit([srs])

    assert len(srs) == len(srs_corrupted)
    assert ~(srs == srs_corrupted).all()


def test_with_edit_incorrect_probabilities():
    with pytest.raises(ValueError) as e:
        with_edit(p_insert=0.3, p_delete=0.3, p_substitute=0.3, p_transpose=0.3)

    assert str(e.value) == "probabilities must sum up to 1.0"


def test_with_phonetic_replacement_table(rng):
    df_phonetic_in_out = pd.read_csv(get_asset_path("phonetic-test.csv"))
    srs_original = df_phonetic_in_out["original"]
    srs_corrupted_expected = df_phonetic_in_out["corrupt"]

    corrupt_phonetic = with_phonetic_replacement_table(
        get_asset_path("homophone-de.csv"), rng=rng
    )
    (srs_corrupted_actual,) = corrupt_phonetic([srs_original])

    assert (srs_corrupted_actual == srs_corrupted_expected).all()


def test_with_cldr_keymap_file(rng):
    srs = pd.Series(["d", "e"])
    corrupt_cldr = with_cldr_keymap_file(get_asset_path("de-t-k0-windows.xml"), rng=rng)
    (srs_corrupted,) = corrupt_cldr([srs])

    assert len(srs) == len(srs_corrupted)
    assert (srs.str.len() == srs_corrupted.str.len()).all()
    assert ~(srs == srs_corrupted).all()

    assert srs_corrupted.iloc[0] in "Decsf"  # neighboring keys of `d`
    assert srs_corrupted.iloc[1] in "E3dwr"  # neighboring keys of `e`


def test_with_cldr_keymap_file_and_charset(rng):
    srs = pd.Series(["4", "e"])
    # create a corruptor that only permits modifications to digits
    corrupt_cldr = with_cldr_keymap_file(
        get_asset_path("de-t-k0-windows.xml"),
        charset=string.digits,
        rng=rng,
    )
    (srs_corrupted,) = corrupt_cldr([srs])

    assert len(srs) == len(srs_corrupted)
    assert (srs.str.len() == srs_corrupted.str.len()).all()

    assert srs_corrupted.iloc[0] in "35"
    assert srs_corrupted.iloc[1] == "e"


def test_with_cldr_keymap_file_no_replacement(rng):
    # this should stay the same since á is not mapped in the keymap
    srs = pd.Series(["á"])
    corrupt_cldr = with_cldr_keymap_file(get_asset_path("de-t-k0-windows.xml"))
    (srs_corrupted,) = corrupt_cldr([srs])

    assert len(srs) == len(srs_corrupted)
    assert (srs.str.len() == srs_corrupted.str.len()).all()
    assert (srs == srs_corrupted).all()


def test_with_replacement_table(rng):
    srs = pd.Series(["k", "5", "2", "1", "g", "q", "l", "i"])
    corrupt_replacement = with_replacement_table(get_asset_path("ocr.csv"))
    (srs_corrupted,) = corrupt_replacement([srs])

    assert len(srs) == len(srs_corrupted)
    assert (srs != srs_corrupted).all()


def test_with_replacement_table_multiple_options(rng):
    # `q` has more than one mapping in the replacement table, so running
    # 100 q's through the corruptor should yield different results
    srs = pd.Series(["q"] * 100)
    corrupt_replacement = with_replacement_table(get_asset_path("ocr.csv"), rng=rng)
    (srs_corrupted,) = corrupt_replacement([srs])

    assert len(srs) == len(srs_corrupted)
    assert (srs != srs_corrupted).all()
    assert len(srs_corrupted.unique()) > 1


def test_permute():
    srs_1, srs_2 = pd.Series(["foo"] * 3), pd.Series(["bar"] * 3)
    corrupt_permute = with_permute()
    srs_1_corrupted, srs_2_corrupted = corrupt_permute([srs_1, srs_2])

    assert (srs_1 != srs_1_corrupted).all()
    assert (srs_1 == srs_2_corrupted).all()
    assert (srs_2 != srs_2_corrupted).all()
    assert (srs_2 == srs_1_corrupted).all()


def test_corrupt_dataframe(rng):
    df = pd.DataFrame(
        data={
            "missing": [""] * 10,
            "cat_typo": list("mf" * 5),
            "weighted_typo_edit": list("mf" * 5),
        }
    )

    corr_missing = with_missing_value(value="", strategy="all")
    corr_edit = with_edit(rng=rng)
    corr_cat = with_categorical_values(
        get_asset_path("freq_table_gender.csv"),
        header=True,
        value_column="gender",
        rng=rng,
    )
    corr_typo = with_cldr_keymap_file(
        get_asset_path("de-t-k0-windows.xml"),
        rng=rng,
    )

    df_corr = corrupt_dataframe(
        df,
        {
            "missing": corr_missing,
            "cat_typo": [corr_cat, corr_typo],
            "weighted_typo_edit": [
                (0.7, corr_typo),
                (0.3, corr_edit),
            ],
        },
        rng=rng,
    )

    assert (df_corr["missing"] == "").all()
    # here it's any because there is a chance some values might go unmodified
    assert (df_corr["cat_typo"] != df["cat_typo"]).any()
    assert (df_corr["weighted_typo_edit"] != df["weighted_typo_edit"]).any()


def test_corrupt_dataframe_incorrect_column():
    df = pd.DataFrame(data={"foo": ["bar", "baz"]})

    with pytest.raises(ValueError) as e:
        corrupt_dataframe(df, {"foobar": with_noop()})

    assert str(e.value) == "column `foobar` does not exist, must be one of `foo`"


def test_corrupt_dataframe_probability_sum_too_high():
    df = pd.DataFrame(data={"foo": ["bar", "baz"]})

    with pytest.raises(ValueError) as e:
        corrupt_dataframe(
            df,
            {
                "foo": [
                    (0.8, with_noop()),
                    (0.3, with_missing_value()),
                ],
            },
        )

    assert str(e.value) == "sum of probabilities may not be higher than 1.0, is 1.1"


def test_corrupt_dataframe_pad_probability():
    df_in = pd.DataFrame(data={"foo": ["a"] * 100})
    df_out = corrupt_dataframe(
        df_in,
        {
            "foo": [
                (0.5, with_missing_value("b", "all")),
            ]
        },
    )

    srs_in = df_in["foo"]
    srs_out = df_out["foo"]

    assert not (srs_in == srs_out).all()
    assert (srs_in == srs_out).any()


def test_corrupt_dataframe_multicolumn():
    df_in = pd.DataFrame(
        data={
            "foo": list("abc"),
            "bar": list("def"),
            "baz": list("ghi"),
        }
    )

    srs_foo = df_in["foo"]
    srs_bar = df_in["bar"]

    df_out = corrupt_dataframe(
        df_in,
        {
            ("foo", "bar"): with_permute(),
            "baz": with_missing_value(strategy="all"),
        },
    )

    srs_foo_corrupted = df_out["foo"]
    srs_bar_corrupted = df_out["bar"]

    assert (srs_foo == srs_bar_corrupted).all()
    assert (srs_bar == srs_foo_corrupted).all()
    assert (df_out["baz"] == "").all()


def test_corrupt_dataframe_multicolumn_noop():
    df_in = pd.DataFrame(
        {
            "foo": ["a"] * 100,
            "bar": ["b"] * 100,
        }
    )

    df_out = corrupt_dataframe(df_in, {("foo", "bar"): [(0.5, with_permute())]})

    assert not (df_out["foo"] == "b").all()
    assert (df_out["foo"] == "b").any()
    assert not (df_out["bar"] == "a").all()
    assert (df_out["bar"] == "a").any()
