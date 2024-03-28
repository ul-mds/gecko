import string

import numpy as np
import pandas as pd
import pytest

from gecko.mutator import (
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
    mutate_data_frame,
    with_noop,
    with_function,
    with_permute,
    Mutator,
)
from tests.helpers import get_asset_path


def test_with_function(rng):
    # basic mutator that simply adds a random number from 0 to 9
    def _mutator(value: str, rand) -> str:
        return value + str(rand.integers(0, 9))

    srs = pd.Series(["foo", "bar", "baz"])
    mutate_ints = with_function(_mutator, rand=rng)
    (srs_mutated,) = mutate_ints([srs])

    for i in range(len(srs)):
        x_orig, x_mut = srs.iloc[i], srs_mutated.iloc[i]

        assert x_orig != x_mut
        assert len(x_mut) == len(x_orig) + 1
        assert x_mut[-1:] in string.digits


def test_with_value_replace_all():
    srs = pd.Series(["foo", "   ", ""])
    mutate_missing = with_missing_value("bar", "all")
    (srs_mutated,) = mutate_missing([srs])

    assert (srs_mutated == pd.Series(["bar", "bar", "bar"])).all()


def test_with_value_replace_empty():
    srs = pd.Series(["foo", "   ", ""])
    mutate_missing = with_missing_value("bar", "empty")
    (srs_mutated,) = mutate_missing([srs])

    assert (srs_mutated == pd.Series(["foo", "   ", "bar"])).all()


def test_with_value_replace_blank():
    srs = pd.Series(["foo", "   ", ""])
    mutate_missing = with_missing_value("bar", "blank")
    (srs_mutated,) = mutate_missing([srs])

    assert (srs_mutated == pd.Series(["foo", "bar", "bar"])).all()


def test_with_random_insert(rng):
    srs = pd.Series(["foo", "bar", "baz"])
    mutate_insert = with_insert(charset="x", rng=rng)
    (srs_mutated,) = mutate_insert([srs])

    # check that series are of the same length
    assert len(srs) == len(srs_mutated)
    # check that all strings are different from one another
    assert ~(srs == srs_mutated).all()

    # check that all string pairs are different in only one char
    for i in range(len(srs)):
        assert len(srs.iloc[i]) + 1 == len(srs_mutated.iloc[i])
        # check that this char is the `x`
        assert "x" not in srs.iloc[i]
        assert "x" in srs_mutated.iloc[i]


def test_with_random_delete(rng):
    srs = pd.Series(["foo", "bar", "baz"])
    mutate_delete = with_delete(rng=rng)
    (srs_mutated,) = mutate_delete([srs])

    # check that series are of the same length
    assert len(srs) == len(srs_mutated)
    # check that all strings are different from one another
    assert ~(srs == srs_mutated).all()
    # check that all string pairs are different in one char
    assert ((srs.str.len() - 1) == srs_mutated.str.len()).all()


def test_with_random_delete_empty_string(rng):
    srs = pd.Series(["", "f"])
    mutate_delete = with_delete(rng=rng)
    (srs_mutated,) = mutate_delete([srs])

    assert len(srs) == len(srs_mutated)
    assert (srs_mutated == "").all()


def test_with_random_transpose(rng):
    srs = pd.Series(["abc", "def", "ghi"])
    mutate_transpose = with_transpose(rng=rng)
    (srs_mutated,) = mutate_transpose([srs])

    # same lengths
    assert len(srs) == len(srs_mutated)
    # all different
    assert ~(srs == srs_mutated).all()
    # same string lengths
    assert (srs.str.len() == srs_mutated.str.len()).all()

    # check that the characters are the same in both series
    for i in range(len(srs)):
        assert set(srs.iloc[i]) == set(srs_mutated.iloc[i])


def test_with_random_transpose_no_neighbor(rng):
    srs = pd.Series(["", "a", "ab"])
    mutate_transpose = with_transpose(rng=rng)
    (srs_mutated,) = mutate_transpose([srs])

    # same lengths
    assert len(srs) == len(srs_mutated)
    # none transposed except last
    assert (srs_mutated == ["", "a", "ba"]).all()


def test_with_random_substitute(rng):
    srs = pd.Series(["foo", "bar", "baz"])
    mutate_substitute = with_substitute(charset="x", rng=rng)
    (srs_mutated,) = mutate_substitute([srs])

    # same len
    assert len(srs) == len(srs_mutated)
    # all different
    assert ~(srs == srs_mutated).all()
    # same string lengths
    assert (srs.str.len() == srs_mutated.str.len()).all()

    # check that original doesn't contain x
    assert (~srs.str.contains("x")).all()
    # check that mutated copy contains x
    assert srs_mutated.str.contains("x").all()


def test_with_random_substitute_empty_string(rng):
    srs = pd.Series(["", "f"])
    mutate_substitute = with_substitute(charset="x", rng=rng)
    (srs_mutated,) = mutate_substitute([srs])

    # same len
    assert len(srs) == len(srs_mutated)
    assert (srs_mutated == ["", "x"]).all()


def test_with_categorical_values(rng):
    def _generate_gender_list():
        nonlocal rng
        return rng.choice(["m", "f", "d", "x"], size=1000)

    mutate_categorical = with_categorical_values(
        get_asset_path("freq_table_gender.csv"),
        value_column="gender",
        rng=rng,
    )

    srs = pd.Series(_generate_gender_list())
    (srs_mutated,) = mutate_categorical([srs])

    # same length
    assert len(srs) == len(srs_mutated)
    # different items
    assert ~(srs == srs_mutated).all()


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
    mutate_edit = with_edit(
        p_insert=0.25,
        p_delete=0.25,
        p_substitute=0.25,
        p_transpose=0.25,
        charset=string.ascii_letters,
        rng=rng,
    )
    (srs_mutated,) = mutate_edit([srs])

    assert len(srs) == len(srs_mutated)
    assert ~(srs == srs_mutated).all()


def test_with_edit_incorrect_probabilities():
    with pytest.raises(ValueError) as e:
        with_edit(p_insert=0.3, p_delete=0.3, p_substitute=0.3, p_transpose=0.3)

    assert str(e.value) == "probabilities must sum up to 1.0"


def test_with_phonetic_replacement_table(rng):
    df_phonetic_in_out = pd.read_csv(get_asset_path("phonetic-test.csv"))
    srs_original = df_phonetic_in_out["original"]
    srs_mutated_expected = df_phonetic_in_out["corrupt"]

    mutate_phonetic = with_phonetic_replacement_table(
        get_asset_path("homophone-de.csv"), rng=rng
    )
    (srs_mutated_actual,) = mutate_phonetic([srs_original])

    assert (srs_mutated_actual == srs_mutated_expected).all()


def test_with_cldr_keymap_file(rng):
    srs = pd.Series(["d", "e"])
    mutate_cldr = with_cldr_keymap_file(get_asset_path("de-t-k0-windows.xml"), rng=rng)
    (srs_mutated,) = mutate_cldr([srs])

    assert len(srs) == len(srs_mutated)
    assert (srs.str.len() == srs_mutated.str.len()).all()
    assert ~(srs == srs_mutated).all()

    assert srs_mutated.iloc[0] in "Decsf"  # neighboring keys of `d`
    assert srs_mutated.iloc[1] in "E3dwr"  # neighboring keys of `e`


def test_with_cldr_keymap_file_and_charset(rng):
    srs = pd.Series(["4", "e"])
    # create a mutator that only permits modifications to digits
    mutate_cldr = with_cldr_keymap_file(
        get_asset_path("de-t-k0-windows.xml"),
        charset=string.digits,
        rng=rng,
    )
    (srs_mutated,) = mutate_cldr([srs])

    assert len(srs) == len(srs_mutated)
    assert (srs.str.len() == srs_mutated.str.len()).all()

    assert srs_mutated.iloc[0] in "35"
    assert srs_mutated.iloc[1] == "e"


def test_with_cldr_keymap_file_no_replacement(rng):
    # this should stay the same since á is not mapped in the keymap
    srs = pd.Series(["á"])
    mutate_cldr = with_cldr_keymap_file(get_asset_path("de-t-k0-windows.xml"))
    (srs_mutated,) = mutate_cldr([srs])

    assert len(srs) == len(srs_mutated)
    assert (srs.str.len() == srs_mutated.str.len()).all()
    assert (srs == srs_mutated).all()


def test_with_replacement_table(rng):
    srs = pd.Series(["k", "5", "2", "1", "g", "q", "l", "i"])
    mutate_replacement = with_replacement_table(get_asset_path("ocr.csv"), rng=rng)
    (srs_mutated,) = mutate_replacement([srs])

    assert len(srs) == len(srs_mutated)
    assert (srs != srs_mutated).all()


def test_with_replacement_table_multiple_options(rng):
    # `q` has more than one mapping in the replacement table, so running
    # 100 q's through the mutator should yield different results
    srs = pd.Series(["q"] * 100)
    mutate_replacement = with_replacement_table(get_asset_path("ocr.csv"), rng=rng)
    (srs_mutated,) = mutate_replacement([srs])

    assert len(srs) == len(srs_mutated)
    assert (srs != srs_mutated).all()
    assert len(srs_mutated.unique()) > 1


def test_permute():
    srs_1, srs_2 = pd.Series(["foo"] * 3), pd.Series(["bar"] * 3)
    mutate_permute = with_permute()
    srs_1_mutated, srs_2_mutated = mutate_permute([srs_1, srs_2])

    assert (srs_1 != srs_1_mutated).all()
    assert (srs_1 == srs_2_mutated).all()
    assert (srs_2 != srs_2_mutated).all()
    assert (srs_2 == srs_1_mutated).all()


def test_permute_more_than_two(rng):
    srs_1, srs_2, srs_3 = (
        pd.Series(["foo"] * 100),
        pd.Series(["bar"] * 100),
        pd.Series(["baz"] * 100),
    )
    mutate_permute = with_permute(rng)
    srs_1_mutated, srs_2_mutated, srs_3_mutated = mutate_permute([srs_1, srs_2, srs_3])

    orig_srs = [srs_1, srs_2, srs_3]
    mut_srs = [srs_1_mutated, srs_2_mutated, srs_3_mutated]

    for i in range(len(orig_srs)):
        for j in range(len(orig_srs)):
            if i == j:
                # if checking series against itself, ensure that not all original entries are copied over
                assert not (orig_srs[i] == mut_srs[j]).all()

            # check that some entries have been shared between the series
            assert (orig_srs[i] == mut_srs[j]).any()


def test_mutate_data_frame_single(rng):
    df = pd.DataFrame({"foo": list(string.ascii_letters)})
    df_mut = mutate_data_frame(
        df,
        {
            "foo": with_missing_value(strategy="all"),
        },
    )

    assert (df_mut["foo"] == "").all()


def test_mutate_data_frame_multiple(rng):
    df = pd.DataFrame({"foo": list(string.ascii_letters)})
    df_mut = mutate_data_frame(
        df,
        {
            "foo": [
                with_missing_value(strategy="all"),
                with_missing_value(value="bar", strategy="all"),
            ]
        },
    )

    assert (df_mut["foo"] == "").any()
    assert (df_mut["foo"] == "bar").any()


def test_mutate_data_frame_single_weighted(rng):
    df = pd.DataFrame({"foo": list(string.ascii_letters)})
    df_mut = mutate_data_frame(df, {"foo": (0.5, with_missing_value(strategy="all"))})

    assert (df_mut["foo"] == "").any()
    assert not (df_mut["foo"] == "").all()


def test_mutate_data_frame_multiple_weighted(rng):
    df = pd.DataFrame({"foo": list(string.ascii_letters)})
    df_mut = mutate_data_frame(
        df,
        {
            "foo": [
                (0.2, with_missing_value(strategy="all")),
                (0.8, with_missing_value("bar", strategy="all")),
            ]
        },
    )

    assert (df_mut["foo"] == "").any()
    assert (df_mut["foo"] == "bar").any()
    assert (df_mut["foo"] == "").sum() < (df_mut["foo"] == "bar").sum()


def test_mutate_data_frame_incorrect_column():
    df = pd.DataFrame(data={"foo": ["bar", "baz"]})

    with pytest.raises(ValueError) as e:
        mutate_data_frame(df, {"foobar": with_noop()})

    assert str(e.value) == "column `foobar` does not exist, must be one of `foo`"


def test_mutate_data_frame_probability_sum_too_high():
    df = pd.DataFrame(data={"foo": ["bar", "baz"]})

    with pytest.raises(ValueError) as e:
        mutate_data_frame(
            df,
            {
                "foo": [
                    (0.8, with_noop()),
                    (0.3, with_missing_value()),
                ],
            },
        )

    assert str(e.value) == "sum of probabilities may not be higher than 1.0, is 1.1"


def test_mutate_data_frame_pad_probability():
    df_in = pd.DataFrame(data={"foo": ["a"] * 100})
    df_out = mutate_data_frame(
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


def test_mutate_data_frame_multicolumn():
    df_in = pd.DataFrame(
        data={
            "foo": list("abc"),
            "bar": list("def"),
            "baz": list("ghi"),
        }
    )

    srs_foo = df_in["foo"]
    srs_bar = df_in["bar"]

    df_out = mutate_data_frame(
        df_in,
        {
            ("foo", "bar"): with_permute(),
            "baz": with_missing_value(strategy="all"),
        },
    )

    srs_foo_mutated = df_out["foo"]
    srs_bar_mutated = df_out["bar"]

    assert (srs_foo == srs_bar_mutated).all()
    assert (srs_bar == srs_foo_mutated).all()
    assert (df_out["baz"] == "").all()


def test_mutate_data_frame_multicolumn_noop():
    df_in = pd.DataFrame(
        {
            "foo": ["a"] * 100,
            "bar": ["b"] * 100,
        }
    )

    df_out = mutate_data_frame(df_in, {("foo", "bar"): [(0.5, with_permute())]})

    assert not (df_out["foo"] == "b").all()
    assert (df_out["foo"] == "b").any()
    assert not (df_out["bar"] == "a").all()
    assert (df_out["bar"] == "a").any()


# dummy rng (shouldn't be used for testing mutator outputs)
__dummy_rng = np.random.default_rng(5432)


@pytest.mark.parametrize(
    "num_srs,func",
    [
        (1, with_noop()),
        (1, with_missing_value("", "all")),
        (1, with_missing_value("", "blank")),
        (1, with_missing_value("", "empty")),
        (1, with_function(lambda s: s.upper())),
        (1, with_insert(rng=__dummy_rng)),
        (1, with_delete(rng=__dummy_rng)),
        (1, with_transpose(rng=__dummy_rng)),
        (1, with_substitute(rng=__dummy_rng)),
        (1, with_edit(rng=__dummy_rng)),
        (
            1,
            with_categorical_values(
                get_asset_path("freq_table_gender.csv"),
                value_column="gender",
                rng=__dummy_rng,
            ),
        ),
        (
            1,
            with_phonetic_replacement_table(
                get_asset_path("homophone-de.csv"), rng=__dummy_rng
            ),
        ),
        (
            1,
            with_cldr_keymap_file(
                get_asset_path("de-t-k0-windows.xml"), rng=__dummy_rng
            ),
        ),
        (1, with_replacement_table(get_asset_path("ocr.csv"), rng=__dummy_rng)),
        (2, with_permute()),
    ],
)
def test_mutator_no_modify(num_srs: int, func: Mutator, rng):
    # ensure that the original series are NEVER modified in the mutators
    def __random_str():
        return "".join(rng.choice(list(string.printable), size=20))

    # create random series and a copy of it
    srs_list_orig = [
        pd.Series([__random_str() for _ in range(100)]) for _ in range(num_srs)
    ]

    srs_list_copy = [srs.copy() for srs in srs_list_orig]

    _ = func(srs_list_orig)

    for i in range(num_srs):
        assert (srs_list_orig[i] == srs_list_copy[i]).all()


def test_mutate_data_frame_no_modify(rng):
    df_orig = pd.DataFrame(
        {
            "upper": list(string.ascii_uppercase),
            "lower": list(string.ascii_lowercase),
        }
    )

    df_copy = df_orig.copy()

    _ = mutate_data_frame(
        df_orig,
        {
            "upper": with_delete(rng=rng),
            "lower": with_insert(rng=rng),
        },
    )

    assert df_orig.equals(df_copy)


# see https://github.com/ul-mds/gecko/issues/33
def test_permute_on_empty_series(rng):
    mutate_permute = with_permute(rng)
    mutate_permute([pd.Series(), pd.Series()])
