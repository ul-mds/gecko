import string

import pandas as pd
import pytest

import gecko.generator
from gecko.mutator import (
    with_cldr_keymap_file,
    PNotMetWarning,
    with_missing_value,
    with_replacement_table,
    with_delete,
    with_insert,
    with_transpose,
    with_substitute,
    with_uppercase,
    with_lowercase,
    with_function,
    with_repeat,
    with_permute,
    with_generator,
    with_group,
    with_categorical_values,
    with_datetime_offset,
)
from tests.helpers import get_asset_path, random_strings, write_temporary_csv_file


def test_with_cldr_keymap_file(rng):
    srs = pd.Series(list(string.ascii_lowercase))
    mutate_cldr = with_cldr_keymap_file(get_asset_path("de-t-k0-windows.xml"), rng=rng)
    (srs_mutated,) = mutate_cldr([srs], 1.0)

    assert len(srs) == len(srs_mutated)
    assert (srs.str.len() == srs_mutated.str.len()).all()
    assert (srs != srs_mutated).all()


def test_with_cldr_keymap_file_multiple_options(rng):
    srs = pd.Series(["foobar"] * 100)
    mutate_cldr = with_cldr_keymap_file(get_asset_path("de-t-k0-windows.xml"), rng=rng)
    (srs_mutated,) = mutate_cldr([srs], 1.0)

    assert len(srs) == len(srs_mutated)
    assert (srs.str.len() == srs_mutated.str.len()).all()
    assert len(srs_mutated.unique()) > 1


def test_with_cldr_keymap_file_warn_low_p(rng):
    # restrain mutator to digits only. if p=0.5 then this should put out a warning.
    srs = pd.Series(["123"] * 20 + ["foobar"] * 80)
    mutate_cldr = with_cldr_keymap_file(
        get_asset_path("de-t-k0-windows.xml"), charset=string.digits, rng=rng
    )

    with pytest.warns(PNotMetWarning) as record:
        (srs_mutated,) = mutate_cldr([srs], 0.5)

    assert len(record) == 1
    assert (
        record[0]
        .message.args[0]
        .startswith("with_cldr_keymap_file: desired probability of 0.5 cannot be met")
    )

    srs_digits, srs_foobar = srs.iloc[:20], srs.iloc[20:]
    srs_mutated_digits, srs_mutated_foobar = (
        srs_mutated.iloc[:20],
        srs_mutated.iloc[20:],
    )

    assert (srs_digits != srs_mutated_digits).all()
    assert (srs_foobar == srs_mutated_foobar).all()


def test_with_missing_value(rng):
    srs = pd.Series(range(1_000), dtype=str)
    mut_missing = with_missing_value(value="", rng=rng)
    (srs_mut,) = mut_missing([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs_mut == "").all()


def test_with_missing_value_existing(rng):
    srs = pd.Series(["foo"] * 20 + [""] * 80)
    mut_missing = with_missing_value(value="", rng=rng)

    with pytest.warns(PNotMetWarning) as record:
        (srs_mut,) = mut_missing([srs], 0.5)

    assert len(record) == 1
    assert (
        record[0]
        .message.args[0]
        .startswith("with_missing_value: desired probability of 0.5 cannot be met")
    )

    assert len(srs) == len(srs_mut)
    assert (srs_mut == "").all()


def test_with_replacement_table(rng):
    srs = pd.Series(list(string.ascii_lowercase))

    # replacement table that maps lowercase chars to uppercase
    df = pd.DataFrame(
        list(zip(string.ascii_lowercase, string.ascii_uppercase)),
        columns=["source", "target"],
    )

    mut_replacement_table = with_replacement_table(
        df, source_column="source", target_column="target", rng=rng
    )

    (srs_mut,) = mut_replacement_table([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() == srs_mut.str.len()).all()


def test_with_replacement_table_reverse(rng):
    # lowercase AND uppercase will be converted to the opposite case
    srs = pd.Series(list(string.ascii_lowercase + string.ascii_uppercase))

    df = pd.DataFrame(
        list(zip(string.ascii_lowercase, string.ascii_uppercase)),
        columns=["source", "target"],
    )

    # this mutator should ensure that uppercase -> lowercase should take place
    mut_replacement_table = with_replacement_table(
        df, source_column="source", target_column="target", reverse=True, rng=rng
    )

    (srs_mut,) = mut_replacement_table([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() == srs_mut.str.len()).all()


def test_with_replacement_table_inline(rng):
    # generate random lowercase strings
    srs = pd.Series(random_strings(str_len=10, charset=string.ascii_lowercase, rng=rng))

    df = pd.DataFrame(
        list(zip(string.ascii_lowercase, string.ascii_uppercase)),
        columns=["source", "target"],
    )

    # this mutator should ensure that replacements occur inline
    mut_replacement_table = with_replacement_table(
        df, source_column="source", target_column="target", inline=True, rng=rng
    )

    (srs_mut,) = mut_replacement_table([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() == srs_mut.str.len()).all()


def test_with_replacement_table_warn_p(rng):
    srs = pd.Series(["a"] * 50 + ["b"] * 50)

    # no mapping for "b"
    df = pd.DataFrame.from_dict({"source": ["a"], "target": ["A"]})

    mut_replacement_table = with_replacement_table(
        df, source_column="source", target_column="target", rng=rng
    )

    with pytest.warns(PNotMetWarning) as record:
        (srs_mut,) = mut_replacement_table([srs], 0.8)

    assert len(record) == 1
    assert (
        record[0]
        .message.args[0]
        .startswith("with_replacement_table: desired probability of 0.8 cannot be met")
    )

    assert len(srs) == len(srs_mut)
    assert (srs.iloc[:50] != srs_mut.iloc[:50]).all()
    assert (srs.iloc[50:] == srs_mut.iloc[50:]).all()
    assert (srs.str.len() == srs_mut.str.len()).all()


def test_with_replacement_table_csv(rng, tmp_path):
    srs = pd.Series(list(string.ascii_lowercase))
    csv_file_path = write_temporary_csv_file(
        tmp_path,
        header=["source", "target"],
        rows=list(zip(string.ascii_lowercase, string.ascii_uppercase)),
    )

    mut_replacement_table = with_replacement_table(
        csv_file_path, source_column="source", target_column="target", rng=rng
    )

    (srs_mut,) = mut_replacement_table([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() == srs_mut.str.len()).all()


def test_with_delete(rng):
    srs = pd.Series(random_strings(rng=rng))
    mut_delete = with_delete(rng=rng)

    (srs_mut,) = mut_delete([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() - 1 == srs_mut.str.len()).all()


def test_with_delete_warn_p(rng):
    srs = pd.Series(random_strings(n_strings=50, rng=rng) + [""] * 50)
    mut_delete = with_delete(rng=rng)

    with pytest.warns(PNotMetWarning) as record:
        (srs_mut,) = mut_delete([srs], 0.8)

    assert len(record) == 1
    assert (
        record[0]
        .message.args[0]
        .startswith("with_delete: desired probability of 0.8 cannot be met")
    )

    assert len(srs) == len(srs_mut)

    assert (srs.iloc[:50] != srs_mut.iloc[:50]).all()
    assert (srs.iloc[:50].str.len() - 1 == srs_mut.iloc[:50].str.len()).all()
    assert (srs.iloc[50:] == srs_mut.iloc[50:]).all()


def test_with_insert(rng):
    srs = pd.Series(random_strings(rng=rng))
    mut_insert = with_insert(rng=rng)

    (srs_mut,) = mut_insert([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() + 1 == srs_mut.str.len()).all()


def test_with_insert_charset(rng):
    # test by inserting uppercase characters into lowercase strings
    srs = pd.Series(random_strings(rng=rng, charset=string.ascii_lowercase))
    mut_insert = with_insert(charset=string.ascii_uppercase, rng=rng)

    (srs_mut,) = mut_insert([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() + 1 == srs_mut.str.len()).all()

    # test that all rows are no longer only lowercase
    assert srs_mut.str.lower().all()
    assert (~srs_mut.str.islower()).all()


def test_with_transpose(rng):
    # create unique strings, otherwise the same characters might be swapped
    srs = pd.Series(random_strings(unique=True, rng=rng))
    mut_transpose = with_transpose(rng=rng)

    (srs_mut,) = mut_transpose([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() == srs_mut.str.len()).all()


def test_with_transpose_warn_p(rng):
    srs = pd.Series(random_strings(n_strings=50, unique=True, rng=rng) + ["a"] * 50)
    mut_transpose = with_transpose(rng=rng)

    with pytest.warns(PNotMetWarning) as record:
        (srs_mut,) = mut_transpose([srs], 0.8)

    assert len(record) == 1
    assert (
        record[0]
        .message.args[0]
        .startswith("with_transpose: desired probability of 0.8 cannot be met")
    )

    assert len(srs) == len(srs_mut)
    assert (srs.str.len() == srs_mut.str.len()).all()
    assert (srs.iloc[:50] != srs_mut.iloc[:50]).all()
    assert (srs.iloc[50:] == srs_mut.iloc[50:]).all()


def test_with_substitute(rng):
    # by default, with_substitute inserts characters, so use digits to avoid replacement of same characters
    srs = pd.Series(random_strings(charset=string.digits, rng=rng))
    mut_substitute = with_substitute(rng=rng)

    (srs_mut,) = mut_substitute([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() == srs_mut.str.len()).all()

    # check that the correct characters have been inserted
    assert srs.str.isdigit().all()
    assert not srs_mut.str.isdigit().all()


def test_with_substitute_charset(rng):
    # same as above, this time using a custom charset param
    srs = pd.Series(random_strings(charset=string.ascii_lowercase, rng=rng))
    mut_substitute = with_substitute(charset=string.digits, rng=rng)

    (srs_mut,) = mut_substitute([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() == srs_mut.str.len()).all()

    # same check as above, this time checking if digits have been inserted
    assert srs.str.isalpha().all()
    assert not srs_mut.str.isalpha().all()
    assert srs_mut.str.isalnum().all()  # should be alphanumeric now!


def test_with_substitute_warn_p(rng):
    srs = pd.Series(
        random_strings(n_strings=50, charset=string.digits, rng=rng) + [""] * 50
    )
    mut_substitute = with_substitute(rng=rng)

    with pytest.warns(PNotMetWarning) as record:
        (srs_mut,) = mut_substitute([srs], 0.8)

    assert len(record) == 1
    assert (
        record[0]
        .message.args[0]
        .startswith("with_substitute: desired probability of 0.8 cannot be met")
    )

    assert len(srs) == len(srs_mut)
    assert (srs.str.len() == srs_mut.str.len()).all()
    assert (srs.iloc[:50] != srs_mut.iloc[:50]).all()
    assert (srs.iloc[50:] == srs_mut.iloc[50:]).all()


def test_with_uppercase(rng):
    srs = pd.Series(random_strings(charset=string.ascii_lowercase, rng=rng))
    mut_uppercase = with_uppercase(rng=rng)

    (srs_mut,) = mut_uppercase([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.upper() == srs_mut).all()


def test_with_uppercase_warn_p(rng):
    srs = pd.Series(
        random_strings(n_strings=50, charset=string.ascii_lowercase, rng=rng)
        + random_strings(n_strings=50, charset=string.ascii_uppercase, rng=rng)
    )
    mut_uppercase = with_uppercase(rng=rng)

    with pytest.warns(PNotMetWarning) as record:
        (srs_mut,) = mut_uppercase([srs], 0.8)

    assert len(record) == 1
    assert (
        record[0]
        .message.args[0]
        .startswith("with_uppercase: desired probability of 0.8 cannot be met")
    )

    assert len(srs) == len(srs_mut)
    assert (srs.iloc[:50] != srs_mut.iloc[:50]).all()
    assert (srs.iloc[:50].str.upper() == srs_mut.iloc[:50]).all()
    assert (srs.iloc[50:] == srs_mut.iloc[50:]).all()


def test_with_lowercase(rng):
    srs = pd.Series(random_strings(charset=string.ascii_uppercase, rng=rng))
    mut_lowercase = with_lowercase(rng=rng)

    (srs_mut,) = mut_lowercase([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.lower() == srs_mut).all()


def test_with_lowercase_warn_p(rng):
    srs = pd.Series(
        random_strings(n_strings=50, charset=string.ascii_uppercase, rng=rng)
        + random_strings(n_strings=50, charset=string.ascii_lowercase, rng=rng)
    )
    mut_lowercase = with_lowercase(rng=rng)

    with pytest.warns(PNotMetWarning) as record:
        (srs_mut,) = mut_lowercase([srs], 0.8)

    assert len(record) == 1
    assert (
        record[0]
        .message.args[0]
        .startswith("with_lowercase: desired probability of 0.8 cannot be met")
    )

    assert len(srs) == len(srs_mut)
    assert (srs.iloc[:50] != srs_mut.iloc[:50]).all()
    assert (srs.iloc[:50].str.lower() == srs_mut.iloc[:50]).all()
    assert (srs.iloc[50:] == srs_mut.iloc[50:]).all()


def test_with_function(rng):
    import numpy as np

    def _my_func(value: str, my_rng: np.random.Generator) -> str:
        return f"{value}{my_rng.integers(0, 10)}"

    srs = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))
    mut_fn = with_function(_my_func, rng=rng, my_rng=rng)

    (srs_mut,) = mut_fn([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs.str.len() + 1 == srs_mut.str.len()).all()
    assert (srs == srs_mut.str[:-1]).all()
    assert srs_mut.str[-1].str.isdigit().all()


def test_with_repeat(rng):
    srs = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))
    mut_repeat = with_repeat(rng=rng)

    (srs_mut,) = mut_repeat([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs + " " + srs == srs_mut).all()


def test_with_repeat_join_character(rng):
    srs = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))
    mut_repeat = with_repeat(join_with=":", rng=rng)

    (srs_mut,) = mut_repeat([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs + ":" + srs == srs_mut).all()


def test_permute(rng):
    srs_a = pd.Series(random_strings(charset=string.ascii_lowercase, rng=rng))
    srs_b = pd.Series(random_strings(charset=string.ascii_uppercase, rng=rng))
    mut_permute = with_permute(rng=rng)

    (srs_a_mut, srs_b_mut) = mut_permute([srs_a, srs_b], 1.0)

    assert len(srs_a) == len(srs_b) == len(srs_a_mut) == len(srs_b_mut)
    assert (srs_a == srs_b_mut).all()
    assert (srs_b == srs_a_mut).all()


def test_permute_multicolumn(rng):
    srs_a = pd.Series(random_strings(charset=string.ascii_lowercase, rng=rng))
    srs_b = pd.Series(random_strings(charset=string.ascii_uppercase, rng=rng))
    srs_c = pd.Series(random_strings(charset=string.digits, rng=rng))
    mut_permute = with_permute(rng=rng)

    (srs_a_mut, srs_b_mut, srs_c_mut) = mut_permute([srs_a, srs_b, srs_c], 1.0)

    assert (
        len(srs_a)
        == len(srs_b)
        == len(srs_c)
        == len(srs_a_mut)
        == len(srs_b_mut)
        == len(srs_c_mut)
    )
    assert (~srs_a_mut.str.islower()).all()
    assert (~srs_b_mut.str.isupper()).all()
    assert (~srs_c_mut.str.isdigit()).all()


def test_permute_partial(rng):
    srs_a = pd.Series(random_strings(charset=string.ascii_lowercase, rng=rng))
    srs_b = pd.Series(random_strings(charset=string.ascii_uppercase, rng=rng))
    mut_permute = with_permute(rng=rng)

    # most entries should be flipped
    (srs_a_mut, srs_b_mut) = mut_permute([srs_a, srs_b], 0.8)

    assert len(srs_a) == len(srs_b) == len(srs_a_mut) == len(srs_b_mut)

    srs_a_flipped = srs_a_mut.str.isupper()
    srs_b_flipped = srs_b_mut.str.islower()

    assert not srs_a_flipped.all()
    assert not srs_b_flipped.all()

    assert srs_a_flipped.sum() > (~srs_a_flipped).sum()
    assert srs_b_flipped.sum() > (~srs_b_flipped).sum()


def _from_scalar_value(column_values) -> gecko.generator.Generator:
    if isinstance(column_values, str):
        column_values = (column_values,)

    if not isinstance(column_values, tuple):
        raise ValueError(
            f"column values must be provided as a string or tuple of strings, is {type(column_values)}"
        )

    def _generate(count: int) -> list[pd.Series]:
        return [pd.Series([value] * count) for value in column_values]

    return _generate


def test_with_generator_replace(rng):
    srs = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))
    mut_generator = with_generator(
        _from_scalar_value("foobar"), mode="replace", rng=rng
    )
    (srs_mut,) = mut_generator([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs_mut == "foobar").all()


def test_with_generator_partial(rng):
    srs = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))
    mut_generator = with_generator(
        _from_scalar_value("foobar"), mode="replace", rng=rng
    )
    (srs_mut,) = mut_generator([srs], 0.8)

    assert len(srs) == len(srs_mut)

    srs_value_mutated = srs_mut == "foobar"

    assert not srs_value_mutated.all()
    assert srs_value_mutated.sum() > (~srs_value_mutated).sum()


def test_with_generator_prepend(rng):
    srs = pd.Series(random_strings(charset=string.digits, rng=rng))
    mut_generator = with_generator(
        _from_scalar_value("foobar"), mode="prepend", rng=rng
    )
    (srs_mut,) = mut_generator([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert srs_mut.str.match(r"foobar \d+").all()


def test_with_generator_append(rng):
    srs = pd.Series(random_strings(charset=string.digits, rng=rng))
    mut_generator = with_generator(_from_scalar_value("foobar"), mode="append", rng=rng)
    (srs_mut,) = mut_generator([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert srs_mut.str.match(r"\d+ foobar").all()


def test_with_generator_prepend_join_char(rng):
    srs = pd.Series(random_strings(charset=string.digits, rng=rng))
    mut_generator = with_generator(
        _from_scalar_value("foobar"), mode="prepend", join_with="-", rng=rng
    )
    (srs_mut,) = mut_generator([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert srs_mut.str.match(r"foobar-\d+").all()


def test_with_generator_append_join_char(rng):
    srs = pd.Series(random_strings(charset=string.digits, rng=rng))
    mut_generator = with_generator(
        _from_scalar_value("foobar"), mode="append", join_with="-", rng=rng
    )
    (srs_mut,) = mut_generator([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert srs_mut.str.match(r"\d+-foobar").all()


def test_with_generator_multi_column(rng):
    srs_a = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))
    srs_b = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))

    mut_generator = with_generator(
        _from_scalar_value(("foobar", "foobaz")), mode="replace", rng=rng
    )
    (srs_a_mut, srs_b_mut) = mut_generator([srs_a, srs_b], 1.0)

    assert len(srs_a) == len(srs_b) == len(srs_a_mut) == len(srs_b_mut)

    assert (srs_a != srs_a_mut).all()
    assert (srs_b != srs_b_mut).all()
    assert (srs_a_mut == "foobar").all()
    assert (srs_b_mut == "foobaz").all()


def test_with_generator_raise_mismatched_columns(rng):
    srs = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))
    mut_generator = with_generator(
        _from_scalar_value(("foobar", "foobaz")), mode="replace", rng=rng
    )

    with pytest.raises(ValueError) as e:
        _ = mut_generator([srs], 1.0)

    assert str(e.value) == (
        "generator must generate as many series as provided to the mutator: "
        "got 2, expected 1"
    )


def test_with_group(rng):
    srs = pd.Series(random_strings(charset=string.ascii_letters, str_len=20, rng=rng))
    mut_group = with_group(
        [
            with_insert(charset=string.digits, rng=rng),
            with_delete(rng=rng),
        ],
        rng=rng,
    )

    (srs_mut,) = mut_group([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() != srs_mut.str.len()).all()
    assert set(srs_mut.str.len().unique()) == {19, 21}


def test_with_group_weighted(rng):
    srs = pd.Series(random_strings(charset=string.ascii_letters, str_len=20, rng=rng))
    mut_group = with_group(
        [
            (0.2, with_insert(charset=string.digits, rng=rng)),
            (0.8, with_delete(rng=rng)),
        ],
        rng=rng,
    )

    (srs_mut,) = mut_group([srs], 1.0)

    assert len(srs) == len(srs_mut)

    srs_mut_str_len_counts = srs_mut.str.len().value_counts()
    assert srs_mut_str_len_counts[19] > srs_mut_str_len_counts[21]


def test_with_group_padded(rng):
    srs = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))
    mut_group = with_group(
        [
            (0.2, with_insert(charset=string.digits, rng=rng)),
        ],
        rng=rng,
    )

    (srs_mut,) = mut_group([srs], 1.0)

    assert len(srs) == len(srs_mut)

    srs_mut_str_len_counts = srs_mut.str.len().value_counts()
    assert srs_mut_str_len_counts[20] > srs_mut_str_len_counts[21]


def test_with_group_raise_p_sum_too_high(rng):
    with pytest.raises(ValueError) as e:
        _ = with_group(
            [
                (0.6, with_delete(rng=rng)),
                (0.41, with_insert(rng=rng)),
            ],
            rng=rng,
        )

    assert str(e.value) == f"sum of weights must not be higher than 1, is {.6 + .41}"


def test_with_group_raise_p_sum_too_low(rng):
    with pytest.raises(ValueError) as e:
        _ = with_group(
            [
                (0, with_delete(rng=rng)),
                (0, with_insert(rng=rng)),
            ],
            rng=rng,
        )

    assert str(e.value) == "sum of weights must be higher than 0, is 0"


def test_with_categorical_values(rng):
    values = {"a", "b", "c", "d"}
    srs = pd.Series(list(sorted(values)) * 1_000)

    df_cat = pd.DataFrame.from_dict({"values": list("abcd")})

    mut_categorical = with_categorical_values(df_cat, value_column="values", rng=rng)

    (srs_mut,) = mut_categorical([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()

    # check that each other categorical value appears at least once
    for value in values:
        this_val = srs == value

        for other_value in values - {value}:
            assert (srs_mut[this_val] == other_value).any()


def test_with_categorical_values_warn_p(rng):
    srs = pd.Series(list("abcd") * 1_000)

    # only two values from the series can be mutated, so p should be approx. 50%
    df_cat = pd.DataFrame.from_dict({"values": list("ab")})

    mut_categorical = with_categorical_values(df_cat, value_column="values", rng=rng)

    with pytest.warns(PNotMetWarning) as record:
        (srs_mut,) = mut_categorical([srs], 0.8)

    assert len(record) == 1
    assert (
        record[0]
        .message.args[0]
        .startswith("with_categorical_values: desired probability of 0.8 cannot be met")
    )

    srs_mutated_rows = (srs == "a") | (srs == "b")

    assert (srs.loc[srs_mutated_rows] != srs_mut.loc[srs_mutated_rows]).all()
    assert (srs.loc[~srs_mutated_rows] == srs_mut.loc[~srs_mutated_rows]).all()


def test_with_categorical_values_raise_too_few_values(rng):
    df_cat = pd.DataFrame.from_dict({"values": list("a")})

    with pytest.raises(ValueError) as e:
        _ = with_categorical_values(df_cat, value_column="values", rng=rng)

    assert str(e.value) == "column must contain at least two unique values, has 1"


def test_with_categorical_values_csv(rng, tmp_path):
    srs = pd.Series(list(string.ascii_letters))
    csv_file_path = write_temporary_csv_file(
        tmp_path,
        header=["values"],
        rows=list(string.ascii_letters),
    )

    mut_categorical = with_categorical_values(
        csv_file_path, value_column="values", rng=rng
    )

    (srs_mut,) = mut_categorical([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()


@pytest.mark.parametrize(
    "unit", ["d", "days", "h", "hours", "m", "minutes", "s", "seconds"]
)
def test_with_datetime_offset(rng, unit):
    srs = pd.Series(
        pd.date_range("2020-01-01", "2021-01-01", freq="h", inclusive="left")
    )
    mut_datetime_offset = with_datetime_offset(5, unit, "%Y-%m-%d %H:%M:%S", rng=rng)
    (srs_mut,) = mut_datetime_offset([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()


@pytest.mark.parametrize(
    "unit", ["d", "days", "h", "hours", "m", "minutes", "s", "seconds"]
)
def test_with_datetime_offset_prevent_wraparound(rng, unit):
    srs = pd.Series(["2020-01-01 00:00:00"] * 1_000)
    mut_datetime_offset = with_datetime_offset(
        5, unit, "%Y-%m-%d %H:%M:%S", prevent_wraparound=True, rng=rng
    )

    with pytest.warns(PNotMetWarning) as record:
        (srs_mut,) = mut_datetime_offset([srs], 1)

    assert len(record) == 1
    assert (
        record[0]
        .message.args[0]
        .startswith("with_datetime_offset: desired probability of 1 cannot be met")
    )

    assert len(srs) == len(srs_mut)
    assert not (srs != srs_mut).all()
    assert (srs == srs_mut).any()


def test_with_datetime_offset_custom_format(rng):
    srs = pd.Series(pd.date_range("2020-01-01", periods=28, freq="D")).dt.strftime(
        "%d.%m.%Y"
    )
    mut_datetime_offset = with_datetime_offset(5, "d", "%d.%m.%Y", rng=rng)
    (srs_mut,) = mut_datetime_offset([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert srs_mut.str.fullmatch(r"\d{2}.\d{2}.\d{4}").all()


def test_with_datetime_offset_raise_invalid_delta(rng):
    with pytest.raises(ValueError) as e:
        _ = with_datetime_offset(0, "d", "%Y-%m-%d", rng=rng)

    assert str(e.value) == "delta must be positive, is 0"
