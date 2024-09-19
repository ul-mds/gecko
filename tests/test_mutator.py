import string

import numpy as np
import pandas as pd
import pytest

from gecko.generator import Generator
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
    with_lowercase,
    with_uppercase,
    with_datetime_offset,
    with_generator,
    with_regex_replacement_table,
    with_repeat,
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
    srs = pd.Series(["Jan", "Jann", "Juan"] * 10)
    mutate_replacement = with_replacement_table(
        get_asset_path("given-name.csv"),
        source_column="source",
        target_column="target",
        rng=rng,
    )
    (srs_mutated,) = mutate_replacement([srs])

    msk_juan = srs == "Juan"

    assert len(srs) == len(srs_mutated)
    assert (srs[~msk_juan] != srs_mutated[~msk_juan]).all()
    assert (srs[msk_juan] == srs_mutated[msk_juan]).all()


def test_with_replacement_table_reverse(rng):
    srs = pd.Series(["Jan", "Jann", "Juan"] * 10)
    mutate_replacement = with_replacement_table(
        get_asset_path("given-name.csv"),
        source_column="source",
        target_column="target",
        reverse=True,
        rng=rng,
    )
    (srs_mutated,) = mutate_replacement([srs])

    assert len(srs) == len(srs_mutated)
    assert (srs != srs_mutated).all()


def test_with_replacement_table_inline(rng):
    srs = pd.Series(["k", "5", "2", "1", "g", "q", "l", "i"])
    mutate_replacement = with_replacement_table(
        get_asset_path("ocr.csv"), inline=True, rng=rng
    )
    (srs_mutated,) = mutate_replacement([srs])

    assert len(srs) == len(srs_mutated)
    assert (srs != srs_mutated).all()


def test_with_replacement_table_inline_multiple_options(rng):
    # `q` has more than one mapping in the replacement table, so running
    # 100 q's through the mutator should yield different results
    srs = pd.Series(["q"] * 100)
    mutate_replacement = with_replacement_table(
        get_asset_path("ocr.csv"), inline=True, rng=rng
    )
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


def test_with_lowercase():
    srs = pd.Series(["Foobar"] * 100)
    mutate_lowercase = with_lowercase()
    (srs_mutated,) = mutate_lowercase([srs])

    assert len(srs) == len(srs_mutated)
    assert (srs != srs_mutated).all()
    assert (srs_mutated == "foobar").all()


def test_with_uppercase():
    srs = pd.Series(["Foobar"] * 100)
    mutate_uppercase = with_uppercase()
    (srs_mutated,) = mutate_uppercase([srs])

    assert len(srs) == len(srs_mutated)
    assert (srs != srs_mutated).all()
    assert (srs_mutated == "FOOBAR").all()


@pytest.mark.parametrize("unit", ["D", "h", "m", "s"])
def test_with_datetime_offset(rng, unit):
    srs = pd.Series(
        pd.date_range("2020-01-01", "2021-01-01", freq="h", inclusive="left")
    )
    mutate_datetime_offset = with_datetime_offset(5, unit, "%Y-%m-%d %H:%M:%S", rng=rng)
    (srs_mutated,) = mutate_datetime_offset([srs])

    assert (srs != srs_mutated).all()


@pytest.mark.parametrize("unit", ["D", "h", "m", "s"])
def test_with_datetime_offset_negative_wraparound(rng, unit):
    srs = pd.Series(["2020-01-01 00:00:00"] * 100)
    mutate_datetime_offset = with_datetime_offset(
        5, unit, "%Y-%m-%d %H:%M:%S", prevent_wraparound=True, rng=rng
    )
    (srs_mutated,) = mutate_datetime_offset([srs])

    assert (srs == srs_mutated).any()


def test_with_datetime_offset_custom_format(rng):
    srs = pd.Series(pd.date_range("2020-01-01", periods=28, freq="D")).dt.strftime(
        "%d.%m.%Y"
    )
    mutate_datetime_offset = with_datetime_offset(5, "D", "%d.%m.%Y", rng=rng)
    (srs_mutated,) = mutate_datetime_offset([srs])

    assert (srs != srs_mutated).all()
    assert srs_mutated.str.fullmatch(r"\d{2}.\d{2}.\d{4}").all()


def test_with_datetime_offset_raise_invalid_format(rng):
    srs = pd.Series(["2024-01-01", "02.01.2024"])
    mutate_datetime_offset = with_datetime_offset(5, "D", "%Y-%m-%d", rng=rng)

    with pytest.raises(ValueError) as e:
        mutate_datetime_offset([srs])

    assert str(e.value).startswith(
        'time data "02.01.2024" doesn\'t match format "%Y-%m-%d"'
    )


def test_with_datetime_offset_raise_nonpositive_delta(rng):
    with pytest.raises(ValueError) as e:
        with_datetime_offset(0, "D", "%Y-%m-%d", rng=rng)

    assert str(e.value) == "delta must be positive, is 0"


def _generate_static(value: str) -> Generator:
    def _generate(count: int) -> list[pd.Series]:
        return [pd.Series([value] * count)]

    return _generate


def test_with_generator_replace():
    srs = pd.Series(["foo"] * 100)
    mutate_generator = with_generator(_generate_static("bar"), "replace")
    (srs_mutated,) = mutate_generator([srs])

    assert (srs_mutated == "bar").all()


def test_with_generator_prepend():
    srs = pd.Series(["foo"] * 100)
    mutate_generator = with_generator(_generate_static("bar"), "prepend")
    (srs_mutated,) = mutate_generator([srs])

    assert (srs_mutated == "bar foo").all()


def test_with_generator_append():
    srs = pd.Series(["foo"] * 100)
    mutate_generator = with_generator(_generate_static("bar"), "append")
    (srs_mutated,) = mutate_generator([srs])

    assert (srs_mutated == "foo bar").all()


def test_with_generator_join_character():
    srs = pd.Series(["foo"] * 100)
    mutate_generator = with_generator(_generate_static("bar"), "append", join_with="-")
    (srs_mutated,) = mutate_generator([srs])

    assert (srs_mutated == "foo-bar").all()


def test_with_generator_slice():
    srs = pd.Series(["foo"] * 100)
    srs_sub = srs.iloc[range(0, len(srs), 3)]
    mutate_generator = with_generator(_generate_static("bar"), "append")
    (srs_mutated,) = mutate_generator([srs_sub])

    assert (srs_mutated == "foo bar").all()


def test_with_generator_raise_series_different_length():
    srs_1, srs_2 = pd.Series(["foo"] * 100), pd.Series(["bar"] * 200)
    mutate_generator = with_generator(_generate_static("bar"), "append")

    with pytest.raises(ValueError) as e:
        mutate_generator([srs_1, srs_2])

    assert str(e.value) == "series do not have the same length"


def test_with_generator_raise_generator_incompatible():
    srs_1, srs_2 = pd.Series(["foo"] * 100), pd.Series(["bar"] * 100)
    mutate_generator = with_generator(_generate_static("bar"), "append")

    with pytest.raises(ValueError) as e:
        mutate_generator([srs_1, srs_2])

    assert (
        str(e.value)
        == "generator must generate as many series as provided to the mutator: got 1, expected 2"
    )


def test_with_generator_raise_unaligned_indices():
    srs_1, srs_2 = (
        pd.Series(["foo"] * 3, index=[0, 1, 2]),
        pd.Series(["bar"] * 3, index=[3, 4, 5]),
    )
    mutate_generator = with_generator(_generate_static("bar"), "append")

    with pytest.raises(ValueError) as e:
        mutate_generator([srs_1, srs_2])

    assert str(e.value) == "indices of input series are not aligned"


def test_with_regex_replacement_table_dob_day_month(rng):
    chunk_size = 10

    srs = pd.Series(
        ["2020-11-30"] * chunk_size
        + ["2020-11-20"] * chunk_size
        + ["2020-11-02"] * chunk_size
        + ["2020-11-10"] * chunk_size
        + ["2020-11-01"] * chunk_size
        + ["2020-01-11"] * chunk_size
        + ["2020-10-11"] * chunk_size
    )

    mutate_regex = with_regex_replacement_table(
        get_asset_path("dob-day-month-flip.csv"), pattern_column="pattern", rng=rng
    )

    (srs_mutated,) = mutate_regex([srs])

    assert (srs_mutated != srs).any()

    for i in range(0, len(srs), chunk_size):
        assert (
            srs_mutated.iloc[i : i + chunk_size] != srs.iloc[i : i + chunk_size]
        ).any()


def test_with_regex_replacement_table_year(rng):
    chunk_size = 5
    srs = pd.Series(
        ["2020-01-10"] * chunk_size
        + ["2030-02-20"] * chunk_size
        + ["2040-01-30"] * chunk_size
    )

    mutate_regex = with_regex_replacement_table(
        get_asset_path("dob-year-flip.csv"), pattern_column="pattern", rng=rng
    )

    (srs_mutated,) = mutate_regex([srs])

    assert (srs_mutated != srs).any()

    for i in range(0, len(srs), chunk_size):
        assert (
            srs_mutated.iloc[i : i + chunk_size] != srs.iloc[i : i + chunk_size]
        ).any()


def test_with_regex_replacement_table_six_nine(rng):
    chunk_size = 10
    srs = pd.Series(
        ["2020-06-06"] * chunk_size
        + ["2020-09-06"] * chunk_size
        + ["2020-06-09"] * chunk_size
        + ["2020-09-09"] * chunk_size
    )

    mutate_regex = with_regex_replacement_table(
        get_asset_path("dob-six-nine.csv"), pattern_column="pattern", rng=rng
    )

    (srs_mutated,) = mutate_regex([srs])

    assert (srs_mutated != srs).any()

    for i in range(0, len(srs), chunk_size):
        assert (
            srs_mutated.iloc[i : i + chunk_size] != srs.iloc[i : i + chunk_size]
        ).any()


def test_with_regex_replacement_table_flags(rng):
    srs = pd.Series(["foobar", "Foobar", "fOoBaR"])
    mutate_regex = with_regex_replacement_table(
        get_asset_path("regex-foobar-case-insensitive.csv"),
        pattern_column="pattern",
        flags_column="flags",
        rng=rng,
    )

    (srs_mutated,) = mutate_regex([srs])

    assert (srs_mutated != srs).any()
    assert (srs_mutated == pd.Series(["foobaz", "Foobaz", "fOoBaz"])).all()


def test_with_repeat():
    srs = pd.Series(["foo"] * 100)
    mutate_repeat = with_repeat()

    (srs_mutated,) = mutate_repeat([srs])
    assert (srs_mutated == "foo foo").all()


def test_with_repeat_custom_join():
    srs = pd.Series(["foo"] * 100)
    mutate_repeat = with_repeat(join_with="")

    (srs_mutated,) = mutate_repeat([srs])
    assert (srs_mutated == "foofoo").all()


def test_mutate_data_frame_single(rng):
    df = pd.DataFrame({"foo": list(string.ascii_letters)})
    df_mut = mutate_data_frame(
        df,
        [
            (
                "foo",
                with_missing_value(strategy="all"),
            )
        ],
    )

    assert (df_mut["foo"] == "").all()


def test_mutate_data_frame_multiple(rng):
    df = pd.DataFrame({"foo": list(string.ascii_letters)})
    df_mut = mutate_data_frame(
        df,
        [
            (
                "foo",
                [
                    with_missing_value(strategy="all"),
                    with_missing_value(value="bar", strategy="all"),
                ],
            )
        ],
    )

    assert (df_mut["foo"] == "").any()
    assert (df_mut["foo"] == "bar").any()


def test_mutate_data_frame_single_weighted(rng):
    df = pd.DataFrame({"foo": list(string.ascii_letters)})
    df_mut = mutate_data_frame(df, [("foo", (0.5, with_missing_value(strategy="all")))])

    assert (df_mut["foo"] == "").any()
    assert not (df_mut["foo"] == "").all()


def test_mutate_data_frame_multiple_weighted(rng):
    df = pd.DataFrame({"foo": list(string.ascii_letters)})
    df_mut = mutate_data_frame(
        df,
        [
            (
                "foo",
                [
                    (0.2, with_missing_value(strategy="all")),
                    (0.8, with_missing_value("bar", strategy="all")),
                ],
            )
        ],
    )

    assert (df_mut["foo"] == "").any()
    assert (df_mut["foo"] == "bar").any()
    assert (df_mut["foo"] == "").sum() < (df_mut["foo"] == "bar").sum()


def test_mutate_data_frame_incorrect_column():
    df = pd.DataFrame(data={"foo": ["bar", "baz"]})

    with pytest.raises(ValueError) as e:
        mutate_data_frame(df, [("foobar", with_noop())])

    assert str(e.value) == "column `foobar` does not exist, must be one of `foo`"


def test_mutate_data_frame_probability_sum_too_high():
    df = pd.DataFrame(data={"foo": ["bar", "baz"]})

    with pytest.raises(ValueError) as e:
        mutate_data_frame(
            df,
            [
                (
                    "foo",
                    [
                        (0.8, with_noop()),
                        (0.3, with_missing_value()),
                    ],
                )
            ],
        )

    assert str(e.value) == "sum of probabilities may not be higher than 1.0, is 1.1"


def test_mutate_data_frame_pad_probability():
    df_in = pd.DataFrame(data={"foo": ["a"] * 100})
    df_out = mutate_data_frame(
        df_in,
        [
            (
                "foo",
                [
                    (0.5, with_missing_value("b", "all")),
                ],
            )
        ],
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
        [(("foo", "bar"), with_permute()), ("baz", with_missing_value(strategy="all"))],
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

    df_out = mutate_data_frame(df_in, [(("foo", "bar"), [(0.5, with_permute())])])

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
        (1, with_lowercase()),
        (1, with_uppercase()),
        (1, with_generator(_generate_static("foo"), "prepend")),
        (1, with_generator(_generate_static("foo"), "append")),
        (1, with_generator(_generate_static("foo"), "replace")),
        (1, with_repeat()),
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
        [
            ("upper", with_delete(rng=rng)),
            ("lower", with_insert(rng=rng)),
        ],
    )

    assert df_orig.equals(df_copy)


# see https://github.com/ul-mds/gecko/issues/33
# this is to test that none of the mutators fail if they are provided an empty series
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
def test_no_error_on_empty_series(num_srs: int, func: Mutator):
    func([pd.Series() for _ in range(num_srs)])


def test_multiple_mutators_per_column(rng):
    df_in = pd.DataFrame(
        {
            "foo": ["a"] * 100,
            "bar": ["b"] * 100,
        }
    )

    df_out = mutate_data_frame(
        df_in,
        [("foo", with_delete(rng=rng)), (("foo", "bar"), with_permute(rng=rng))],
        rng=rng,
    )

    # check that permutation and deletion was applied (`a` should no longer be present)
    assert (df_out["foo"] == "b").all()
    assert (df_out["bar"] == "").all()


# see https://github.com/ul-mds/gecko/issues/41
@pytest.mark.parametrize(
    "value,value_type",
    [
        (1, int),
        (1.0, float),
    ],
)
def test_mutate_data_frame_numeric_input(value, value_type, rng):
    # sanity check
    assert isinstance(value, value_type)

    df_in = pd.DataFrame({"foo": ["a"] * 100})

    # before the fix mentioned above, this failed if the provided probability was an int
    _ = mutate_data_frame(df_in, [("foo", (value, with_delete(rng=rng)))], rng=rng)


def test_mutate_data_frame_order_generates_different_results(rng_factory):
    rng_1 = rng_factory()
    rng_2 = rng_factory()

    df_in = pd.DataFrame({"foo": ["a"] * 100})

    df_out_1 = mutate_data_frame(
        df_in,
        [
            ("foo", (0.5, with_delete(rng=rng_1))),
            ("foo", (0.5, with_insert(rng=rng_1))),
        ],
        rng=rng_1,
    )

    df_out_2 = mutate_data_frame(
        df_in,
        [
            ("foo", (0.5, with_insert(rng=rng_2))),
            ("foo", (0.5, with_delete(rng=rng_2))),
        ],
        rng=rng_2,
    )

    assert not (df_out_1["foo"] == df_out_2["foo"]).all()


# see https://github.com/ul-mds/gecko/issues/69
# CSV files containing empty cells will be evaluated to NaN, which causes this to raise an error
def test_with_replacement_table_nan(tmp_path, rng):
    replacement_table_file_path = tmp_path / "replacement.csv"
    replacement_table_file_path.write_text('source,target\n"-",""\n')

    mut_replace = with_replacement_table(
        replacement_table_file_path,
        source_column="source",
        target_column="target",
        inline=True,
        rng=rng,
    )

    srs = pd.Series(["foo-bar", "foo-baz", "foo-bat"])
    (srs_mut,) = mut_replace([srs])

    assert (srs_mut == ["foobar", "foobaz", "foobat"]).all()
