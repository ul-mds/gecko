import itertools
import string

import pandas as pd
import pytest

from gecko import Generator, mutator, GeckoWarning
from tests.helpers import get_asset_path, random_strings, write_temporary_csv_file


def test_with_cldr_keymap_file(rng):
    srs = pd.Series(list(string.ascii_lowercase))
    mutate_cldr = mutator.with_cldr_keymap_file(get_asset_path("de-t-k0-windows.xml"), rng=rng)
    (srs_mutated,) = mutate_cldr([srs], 1.0)

    assert len(srs) == len(srs_mutated)
    assert (srs.str.len() == srs_mutated.str.len()).all()
    assert (srs != srs_mutated).all()


def test_with_cldr_keymap_file_partial(rng):
    srs = pd.Series(list(string.ascii_lowercase))
    mutate_cldr = mutator.with_cldr_keymap_file(get_asset_path("de-t-k0-windows.xml"), rng=rng)
    (srs_mutated,) = mutate_cldr([srs], 0.5)

    assert len(srs) == len(srs_mutated)
    assert (srs == srs_mutated).any()
    assert (srs != srs_mutated).any()


def test_with_cldr_keymap_file_multiple_options(rng):
    srs = pd.Series(["foobar"] * 100)
    mutate_cldr = mutator.with_cldr_keymap_file(get_asset_path("de-t-k0-windows.xml"), rng=rng)
    (srs_mutated,) = mutate_cldr([srs], 1.0)

    assert len(srs) == len(srs_mutated)
    assert (srs.str.len() == srs_mutated.str.len()).all()
    assert len(srs_mutated.unique()) > 1


def test_with_cldr_keymap_file_warn_low_p(rng):
    # restrain mutator to digits only. if p=0.5 then this should put out a warning.
    srs = pd.Series(["123"] * 20 + ["foobar"] * 80)
    mutate_cldr = mutator.with_cldr_keymap_file(get_asset_path("de-t-k0-windows.xml"), charset=string.digits, rng=rng)

    with pytest.warns(GeckoWarning) as record:
        (srs_mutated,) = mutate_cldr([srs], 0.5)

    assert len(record) == 1
    assert record[0].message.args[0].startswith("with_cldr_keymap_file: desired probability of 0.5 cannot be met")

    srs_digits, srs_foobar = srs.iloc[:20], srs.iloc[20:]
    srs_mutated_digits, srs_mutated_foobar = (
        srs_mutated.iloc[:20],
        srs_mutated.iloc[20:],
    )

    assert (srs_digits != srs_mutated_digits).all()
    assert (srs_foobar == srs_mutated_foobar).all()


def test_with_missing_value(rng):
    srs = pd.Series(range(1_000), dtype=str)
    mut_missing = mutator.with_missing_value(value="", rng=rng)
    (srs_mut,) = mut_missing([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs_mut == "").all()


def test_with_missing_value_partial(rng):
    srs = pd.Series(range(1_000), dtype=str)
    mut_missing = mutator.with_missing_value(value="", rng=rng)
    (srs_mut,) = mut_missing([srs], 0.5)

    assert len(srs) == len(srs_mut)
    assert (srs == srs_mut).any()
    assert (srs != srs_mut).any()


def test_with_missing_value_existing(rng):
    srs = pd.Series(["foo"] * 20 + [""] * 80)
    mut_missing = mutator.with_missing_value(value="", rng=rng)

    with pytest.warns(GeckoWarning) as record:
        (srs_mut,) = mut_missing([srs], 0.5)

    assert len(record) == 1
    assert record[0].message.args[0].startswith("with_missing_value: desired probability of 0.5 cannot be met")

    assert len(srs) == len(srs_mut)
    assert (srs_mut == "").all()


def test_with_replacement_table(rng):
    srs = pd.Series(list(string.ascii_lowercase))

    # replacement table that maps lowercase chars to uppercase
    df = pd.DataFrame(
        list(zip(string.ascii_lowercase, string.ascii_uppercase)),
        columns=["source", "target"],
    )

    mut_replacement_table = mutator.with_replacement_table(df, source_column="source", target_column="target", rng=rng)

    (srs_mut,) = mut_replacement_table([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() == srs_mut.str.len()).all()


def test_with_replacement_table_random_values(rng):
    srs = pd.Series(["aaa"] * 1_000)
    df_replacement_table = pd.DataFrame.from_dict({"source": ["a", "a", "a"], "target": ["0", "1", "2"]})

    mut_replacement_table = mutator.with_replacement_table(
        df_replacement_table,
        source_column="source",
        target_column="target",
        inline=True,
        rng=rng,
    )

    (srs_mut,) = mut_replacement_table([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert len(srs_mut.unique()) > 1


def test_with_replacement_table_favor_rare_replacements(rng):
    srs = pd.Series(["foobar"] * 100 + ["foobaz"] * 50)

    df = pd.DataFrame.from_dict({"source": ["foobar", "foobaz"], "target": ["0", "1"]})

    mut_replacement_table = mutator.with_replacement_table(df, source_column="source", target_column="target", rng=rng)

    (srs_mutated,) = mut_replacement_table([srs], 1)

    assert len(srs) == len(srs_mutated)
    assert (srs != srs_mutated).all()
    assert (srs_mutated == ["0"] * 100 + ["1"] * 50).all()


def test_with_replacement_table_partial(rng):
    srs = pd.Series(list(string.ascii_lowercase))

    # replacement table that maps lowercase chars to uppercase
    df = pd.DataFrame(
        list(zip(string.ascii_lowercase, string.ascii_uppercase)),
        columns=["source", "target"],
    )

    mut_replacement_table = mutator.with_replacement_table(df, source_column="source", target_column="target", rng=rng)

    (srs_mut,) = mut_replacement_table([srs], 0.5)

    assert len(srs) == len(srs_mut)
    assert (srs == srs_mut).any()
    assert (srs != srs_mut).any()


def test_with_replacement_table_reverse(rng):
    # lowercase AND uppercase will be converted to the opposite case
    srs = pd.Series(list(string.ascii_lowercase + string.ascii_uppercase))

    df = pd.DataFrame(
        list(zip(string.ascii_lowercase, string.ascii_uppercase)),
        columns=["source", "target"],
    )

    # this mutator should ensure that uppercase -> lowercase should take place
    mut_replacement_table = mutator.with_replacement_table(
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
    mut_replacement_table = mutator.with_replacement_table(
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

    mut_replacement_table = mutator.with_replacement_table(df, source_column="source", target_column="target", rng=rng)

    with pytest.warns(GeckoWarning) as record:
        (srs_mut,) = mut_replacement_table([srs], 0.8)

    assert len(record) == 1
    assert record[0].message.args[0].startswith("with_replacement_table: desired probability of 0.8 cannot be met")

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

    mut_replacement_table = mutator.with_replacement_table(
        csv_file_path, source_column="source", target_column="target", rng=rng
    )

    (srs_mut,) = mut_replacement_table([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() == srs_mut.str.len()).all()


def test_with_delete(rng):
    srs = pd.Series(random_strings(rng=rng))
    mut_delete = mutator.with_delete(rng=rng)

    (srs_mut,) = mut_delete([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() - 1 == srs_mut.str.len()).all()


def test_with_delete_partial(rng):
    srs = pd.Series(random_strings(rng=rng))
    mut_delete = mutator.with_delete(rng=rng)

    (srs_mut,) = mut_delete([srs], 0.5)

    assert len(srs) == len(srs_mut)
    assert (srs == srs_mut).any()
    assert (srs != srs_mut).any()


def test_with_delete_warn_p(rng):
    srs = pd.Series(random_strings(n_strings=50, rng=rng) + [""] * 50)
    mut_delete = mutator.with_delete(rng=rng)

    with pytest.warns(GeckoWarning) as record:
        (srs_mut,) = mut_delete([srs], 0.8)

    assert len(record) == 1
    assert record[0].message.args[0].startswith("with_delete: desired probability of 0.8 cannot be met")

    assert len(srs) == len(srs_mut)

    assert (srs.iloc[:50] != srs_mut.iloc[:50]).all()
    assert (srs.iloc[:50].str.len() - 1 == srs_mut.iloc[:50].str.len()).all()
    assert (srs.iloc[50:] == srs_mut.iloc[50:]).all()


def test_with_insert(rng):
    srs = pd.Series(random_strings(rng=rng))
    mut_insert = mutator.with_insert(rng=rng)

    (srs_mut,) = mut_insert([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() + 1 == srs_mut.str.len()).all()


def test_with_insert_partial(rng):
    srs = pd.Series(random_strings(rng=rng))
    mut_insert = mutator.with_insert(rng=rng)

    (srs_mut,) = mut_insert([srs], 0.5)

    assert len(srs) == len(srs_mut)
    assert (srs == srs_mut).any()
    assert (srs != srs_mut).any()


def test_with_insert_charset(rng):
    # test by inserting uppercase characters into lowercase strings
    srs = pd.Series(random_strings(rng=rng, charset=string.ascii_lowercase))
    mut_insert = mutator.with_insert(charset=string.ascii_uppercase, rng=rng)

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
    mut_transpose = mutator.with_transpose(rng=rng)

    (srs_mut,) = mut_transpose([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() == srs_mut.str.len()).all()


def test_with_transpose_partial(rng):
    # create unique strings, otherwise the same characters might be swapped
    srs = pd.Series(random_strings(unique=True, rng=rng))
    mut_transpose = mutator.with_transpose(rng=rng)

    (srs_mut,) = mut_transpose([srs], 0.5)

    assert len(srs) == len(srs_mut)
    assert (srs == srs_mut).any()
    assert (srs != srs_mut).any()


def test_with_transpose_warn_p(rng):
    srs = pd.Series(random_strings(n_strings=50, unique=True, rng=rng) + ["a"] * 50)
    mut_transpose = mutator.with_transpose(rng=rng)

    with pytest.warns(GeckoWarning) as record:
        (srs_mut,) = mut_transpose([srs], 0.8)

    assert len(record) == 1
    assert record[0].message.args[0].startswith("with_transpose: desired probability of 0.8 cannot be met")

    assert len(srs) == len(srs_mut)
    assert (srs.str.len() == srs_mut.str.len()).all()
    assert (srs.iloc[:50] != srs_mut.iloc[:50]).all()
    assert (srs.iloc[50:] == srs_mut.iloc[50:]).all()


def test_with_substitute(rng):
    # by default, with_substitute inserts characters, so use digits to avoid replacement of same characters
    srs = pd.Series(random_strings(charset=string.digits, rng=rng))
    mut_substitute = mutator.with_substitute(rng=rng)

    (srs_mut,) = mut_substitute([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() == srs_mut.str.len()).all()

    # check that the correct characters have been inserted
    assert srs.str.isdigit().all()
    assert not srs_mut.str.isdigit().all()


def test_with_substitute_partial(rng):
    srs = pd.Series(random_strings(charset=string.digits, rng=rng))
    mut_substitute = mutator.with_substitute(rng=rng)

    (srs_mut,) = mut_substitute([srs], 0.5)

    assert len(srs) == len(srs_mut)
    assert (srs == srs_mut).any()
    assert (srs != srs_mut).any()


def test_with_substitute_charset(rng):
    # same as above, this time using a custom charset param
    srs = pd.Series(random_strings(charset=string.ascii_lowercase, rng=rng))
    mut_substitute = mutator.with_substitute(charset=string.digits, rng=rng)

    (srs_mut,) = mut_substitute([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() == srs_mut.str.len()).all()

    # same check as above, this time checking if digits have been inserted
    assert srs.str.isalpha().all()
    assert not srs_mut.str.isalpha().all()
    assert srs_mut.str.isalnum().all()  # should be alphanumeric now!


def test_with_substitute_warn_p(rng):
    srs = pd.Series(random_strings(n_strings=50, charset=string.digits, rng=rng) + [""] * 50)
    mut_substitute = mutator.with_substitute(rng=rng)

    with pytest.warns(GeckoWarning) as record:
        (srs_mut,) = mut_substitute([srs], 0.8)

    assert len(record) == 1
    assert record[0].message.args[0].startswith("with_substitute: desired probability of 0.8 cannot be met")

    assert len(srs) == len(srs_mut)
    assert (srs.str.len() == srs_mut.str.len()).all()
    assert (srs.iloc[:50] != srs_mut.iloc[:50]).all()
    assert (srs.iloc[50:] == srs_mut.iloc[50:]).all()


def test_with_uppercase(rng):
    srs = pd.Series(random_strings(charset=string.ascii_lowercase, rng=rng))
    mut_uppercase = mutator.with_uppercase(rng=rng)

    (srs_mut,) = mut_uppercase([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.upper() == srs_mut).all()


def test_with_uppercase_partial(rng):
    srs = pd.Series(random_strings(charset=string.ascii_lowercase, rng=rng))
    mut_uppercase = mutator.with_uppercase(rng=rng)

    (srs_mut,) = mut_uppercase([srs], 0.5)

    assert len(srs) == len(srs_mut)
    assert (srs == srs_mut).any()
    assert (srs != srs_mut).any()


def test_with_uppercase_warn_p(rng):
    srs = pd.Series(
        random_strings(n_strings=50, charset=string.ascii_lowercase, rng=rng)
        + random_strings(n_strings=50, charset=string.ascii_uppercase, rng=rng)
    )
    mut_uppercase = mutator.with_uppercase(rng=rng)

    with pytest.warns(GeckoWarning) as record:
        (srs_mut,) = mut_uppercase([srs], 0.8)

    assert len(record) == 1
    assert record[0].message.args[0].startswith("with_uppercase: desired probability of 0.8 cannot be met")

    assert len(srs) == len(srs_mut)
    assert (srs.iloc[:50] != srs_mut.iloc[:50]).all()
    assert (srs.iloc[:50].str.upper() == srs_mut.iloc[:50]).all()
    assert (srs.iloc[50:] == srs_mut.iloc[50:]).all()


def test_with_lowercase(rng):
    srs = pd.Series(random_strings(charset=string.ascii_uppercase, rng=rng))
    mut_lowercase = mutator.with_lowercase(rng=rng)

    (srs_mut,) = mut_lowercase([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.lower() == srs_mut).all()


def test_with_lowercase_partial(rng):
    srs = pd.Series(random_strings(charset=string.ascii_uppercase, rng=rng))
    mut_lowercase = mutator.with_lowercase(rng=rng)

    (srs_mut,) = mut_lowercase([srs], 0.5)

    assert len(srs) == len(srs_mut)
    assert (srs == srs_mut).any()
    assert (srs != srs_mut).any()


def test_with_lowercase_warn_p(rng):
    srs = pd.Series(
        random_strings(n_strings=50, charset=string.ascii_uppercase, rng=rng)
        + random_strings(n_strings=50, charset=string.ascii_lowercase, rng=rng)
    )
    mut_lowercase = mutator.with_lowercase(rng=rng)

    with pytest.warns(GeckoWarning) as record:
        (srs_mut,) = mut_lowercase([srs], 0.8)

    assert len(record) == 1
    assert record[0].message.args[0].startswith("with_lowercase: desired probability of 0.8 cannot be met")

    assert len(srs) == len(srs_mut)
    assert (srs.iloc[:50] != srs_mut.iloc[:50]).all()
    assert (srs.iloc[:50].str.lower() == srs_mut.iloc[:50]).all()
    assert (srs.iloc[50:] == srs_mut.iloc[50:]).all()


def test_with_function(rng):
    import numpy as np

    def _my_func(value: str, my_rng: np.random.Generator) -> str:
        return f"{value}{my_rng.integers(0, 10)}"

    srs = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))
    mut_fn = mutator.with_function(_my_func, rng=rng, my_rng=rng)

    (srs_mut,) = mut_fn([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() + 1 == srs_mut.str.len()).all()
    assert (srs == srs_mut.str[:-1]).all()
    assert srs_mut.str[-1].str.isdigit().all()


def test_with_function_partial(rng):
    import numpy as np

    def _my_func(value: str, my_rng: np.random.Generator) -> str:
        return f"{value}{my_rng.integers(0, 10)}"

    srs = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))
    mut_fn = mutator.with_function(_my_func, rng=rng, my_rng=rng)

    (srs_mut,) = mut_fn([srs], 0.5)

    assert len(srs) == len(srs_mut)
    assert (srs == srs_mut).any()
    assert (srs != srs_mut).any()


def test_with_repeat(rng):
    srs = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))
    mut_repeat = mutator.with_repeat(rng=rng)

    (srs_mut,) = mut_repeat([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs + " " + srs == srs_mut).all()


def test_with_repeat_partial(rng):
    srs = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))
    mut_repeat = mutator.with_repeat(rng=rng)

    (srs_mut,) = mut_repeat([srs], 0.5)

    assert len(srs) == len(srs_mut)
    assert (srs == srs_mut).any()
    assert (srs != srs_mut).any()


def test_with_repeat_join_character(rng):
    srs = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))
    mut_repeat = mutator.with_repeat(join_with=":", rng=rng)

    (srs_mut,) = mut_repeat([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs + ":" + srs == srs_mut).all()


def test_with_permute(rng):
    srs_a = pd.Series(random_strings(charset=string.ascii_lowercase, rng=rng))
    srs_b = pd.Series(random_strings(charset=string.ascii_uppercase, rng=rng))
    mut_permute = mutator.with_permute(rng=rng)

    (srs_a_mut, srs_b_mut) = mut_permute([srs_a, srs_b], 1.0)

    assert len(srs_a) == len(srs_b) == len(srs_a_mut) == len(srs_b_mut)
    assert (srs_a == srs_b_mut).all()
    assert (srs_b == srs_a_mut).all()


def test_with_permute_multicolumn(rng):
    srs_a = pd.Series(random_strings(charset=string.ascii_lowercase, rng=rng))
    srs_b = pd.Series(random_strings(charset=string.ascii_uppercase, rng=rng))
    srs_c = pd.Series(random_strings(charset=string.digits, rng=rng))
    mut_permute = mutator.with_permute(rng=rng)

    (srs_a_mut, srs_b_mut, srs_c_mut) = mut_permute([srs_a, srs_b, srs_c], 1.0)

    assert len(srs_a) == len(srs_b) == len(srs_c) == len(srs_a_mut) == len(srs_b_mut) == len(srs_c_mut)
    assert (~srs_a_mut.str.islower()).all()
    assert (~srs_b_mut.str.isupper()).all()
    assert (~srs_c_mut.str.isdigit()).all()


def test_with_permute_partial(rng):
    srs_a = pd.Series(random_strings(charset=string.ascii_lowercase, rng=rng))
    srs_b = pd.Series(random_strings(charset=string.ascii_uppercase, rng=rng))
    mut_permute = mutator.with_permute(rng=rng)

    # most entries should be flipped
    (srs_a_mut, srs_b_mut) = mut_permute([srs_a, srs_b], 0.8)

    assert len(srs_a) == len(srs_b) == len(srs_a_mut) == len(srs_b_mut)

    srs_a_flipped = srs_a_mut.str.isupper()
    srs_b_flipped = srs_b_mut.str.islower()

    assert not srs_a_flipped.all()
    assert not srs_b_flipped.all()

    assert srs_a_flipped.sum() > (~srs_a_flipped).sum()
    assert srs_b_flipped.sum() > (~srs_b_flipped).sum()


def _from_scalar_value(column_values) -> Generator:
    if isinstance(column_values, str):
        column_values = (column_values,)

    if not isinstance(column_values, tuple):
        raise ValueError(f"column values must be provided as a string or tuple of strings, is {type(column_values)}")

    def _generate(count: int) -> list[pd.Series]:
        return [pd.Series([value] * count) for value in column_values]

    return _generate


def test_with_generator_replace(rng):
    srs = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))
    mut_generator = mutator.with_generator(_from_scalar_value("foobar"), mode="replace", rng=rng)
    (srs_mut,) = mut_generator([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs_mut == "foobar").all()


def test_with_generator_partial(rng):
    srs = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))
    mut_generator = mutator.with_generator(_from_scalar_value("foobar"), mode="replace", rng=rng)
    (srs_mut,) = mut_generator([srs], 0.8)

    assert len(srs) == len(srs_mut)

    srs_value_mutated = srs_mut == "foobar"

    assert not srs_value_mutated.all()
    assert srs_value_mutated.sum() > (~srs_value_mutated).sum()


def test_with_generator_prepend(rng):
    srs = pd.Series(random_strings(charset=string.digits, rng=rng))
    mut_generator = mutator.with_generator(_from_scalar_value("foobar"), mode="prepend", rng=rng)
    (srs_mut,) = mut_generator([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert srs_mut.str.match(r"foobar \d+").all()


def test_with_generator_append(rng):
    srs = pd.Series(random_strings(charset=string.digits, rng=rng))
    mut_generator = mutator.with_generator(_from_scalar_value("foobar"), mode="append", rng=rng)
    (srs_mut,) = mut_generator([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert srs_mut.str.match(r"\d+ foobar").all()


def test_with_generator_prepend_join_char(rng):
    srs = pd.Series(random_strings(charset=string.digits, rng=rng))
    mut_generator = mutator.with_generator(_from_scalar_value("foobar"), mode="prepend", join_with="-", rng=rng)
    (srs_mut,) = mut_generator([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert srs_mut.str.match(r"foobar-\d+").all()


def test_with_generator_append_join_char(rng):
    srs = pd.Series(random_strings(charset=string.digits, rng=rng))
    mut_generator = mutator.with_generator(_from_scalar_value("foobar"), mode="append", join_with="-", rng=rng)
    (srs_mut,) = mut_generator([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert srs_mut.str.match(r"\d+-foobar").all()


def test_with_generator_prepend_join_char_insert(rng):
    srs = pd.Series(random_strings(charset=string.digits, rng=rng))
    mut_generator = mutator.with_generator(_from_scalar_value("foobar"), mode="prepend", join_with=" ({}) ", rng=rng)
    (srs_mut,) = mut_generator([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert srs_mut.str.match(r" \(foobar\) \d+").all()


def test_with_generator_append_join_char_insert(rng):
    srs = pd.Series(random_strings(charset=string.digits, rng=rng))
    mut_generator = mutator.with_generator(_from_scalar_value("foobar"), mode="append", join_with=" ({}) ", rng=rng)
    (srs_mut,) = mut_generator([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert srs_mut.str.match(r"\d+ \(foobar\) ").all()


def test_with_generator_multi_column(rng):
    srs_a = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))
    srs_b = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))

    mut_generator = mutator.with_generator(_from_scalar_value(("foobar", "foobaz")), mode="replace", rng=rng)
    (srs_a_mut, srs_b_mut) = mut_generator([srs_a, srs_b], 1.0)

    assert len(srs_a) == len(srs_b) == len(srs_a_mut) == len(srs_b_mut)

    assert (srs_a != srs_a_mut).all()
    assert (srs_b != srs_b_mut).all()
    assert (srs_a_mut == "foobar").all()
    assert (srs_b_mut == "foobaz").all()


def test_with_generator_raise_mismatched_columns(rng):
    srs = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))
    mut_generator = mutator.with_generator(_from_scalar_value(("foobar", "foobaz")), mode="replace", rng=rng)

    with pytest.raises(ValueError) as e:
        _ = mut_generator([srs], 1.0)

    assert str(e.value) == ("generator must generate as many series as provided to the mutator: " "got 2, expected 1")


def test_with_group(rng):
    srs = pd.Series(random_strings(charset=string.ascii_letters, str_len=20, rng=rng))
    mut_group = mutator.with_group(
        [
            mutator.with_insert(charset=string.digits, rng=rng),
            mutator.with_delete(rng=rng),
        ],
        rng=rng,
    )

    (srs_mut,) = mut_group([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs.str.len() != srs_mut.str.len()).all()
    assert set(srs_mut.str.len().unique()) == {19, 21}


def test_with_group_partial(rng):
    srs = pd.Series(random_strings(charset=string.ascii_letters, str_len=20, rng=rng))
    mut_group = mutator.with_group(
        [
            mutator.with_insert(charset=string.digits, rng=rng),
            mutator.with_delete(rng=rng),
        ],
        rng=rng,
    )

    (srs_mut,) = mut_group([srs], 0.5)

    assert len(srs) == len(srs_mut)
    assert (srs == srs_mut).any()
    assert (srs != srs_mut).any()


def test_with_group_weighted(rng):
    srs = pd.Series(random_strings(charset=string.ascii_letters, str_len=20, rng=rng))
    mut_group = mutator.with_group(
        [
            (0.2, mutator.with_insert(charset=string.digits, rng=rng)),
            (0.8, mutator.with_delete(rng=rng)),
        ],
        rng=rng,
    )

    (srs_mut,) = mut_group([srs], 1.0)

    assert len(srs) == len(srs_mut)

    srs_mut_str_len_counts = srs_mut.str.len().value_counts()
    assert srs_mut_str_len_counts[19] > srs_mut_str_len_counts[21]


def test_with_group_padded(rng):
    srs = pd.Series(random_strings(charset=string.ascii_letters, rng=rng))
    mut_group = mutator.with_group(
        [
            (0.2, mutator.with_insert(charset=string.digits, rng=rng)),
        ],
        rng=rng,
    )

    (srs_mut,) = mut_group([srs], 1.0)

    assert len(srs) == len(srs_mut)

    srs_mut_str_len_counts = srs_mut.str.len().value_counts()
    assert srs_mut_str_len_counts[20] > srs_mut_str_len_counts[21]


def test_with_group_raise_p_sum_too_high(rng):
    with pytest.raises(ValueError) as e:
        _ = mutator.with_group(
            [
                (0.6, mutator.with_delete(rng=rng)),
                (0.41, mutator.with_insert(rng=rng)),
            ],
            rng=rng,
        )

    assert str(e.value) == f"sum of weights must not be higher than 1, is {.6 + .41}"


def test_with_group_raise_p_sum_too_low(rng):
    with pytest.raises(ValueError) as e:
        _ = mutator.with_group(
            [
                (0, mutator.with_delete(rng=rng)),
                (0, mutator.with_insert(rng=rng)),
            ],
            rng=rng,
        )

    assert str(e.value) == "sum of weights must be higher than 0, is 0"


def test_with_categorical_values(rng):
    values = {"a", "b", "c", "d"}
    srs = pd.Series(list(sorted(values)) * 1_000)

    df_cat = pd.DataFrame.from_dict({"values": list("abcd")})

    mut_categorical = mutator.with_categorical_values(df_cat, value_column="values", rng=rng)

    (srs_mut,) = mut_categorical([srs], 1.0)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()

    # check that each other categorical value appears at least once
    for value in values:
        this_val = srs == value

        for other_value in values - {value}:
            assert (srs_mut[this_val] == other_value).any()


def test_with_categorical_values_partial(rng):
    values = {"a", "b", "c", "d"}
    srs = pd.Series(list(sorted(values)) * 1_000)

    df_cat = pd.DataFrame.from_dict({"values": list("abcd")})

    mut_categorical = mutator.with_categorical_values(df_cat, value_column="values", rng=rng)

    (srs_mut,) = mut_categorical([srs], 0.5)

    assert len(srs) == len(srs_mut)
    assert (srs == srs_mut).any()
    assert (srs != srs_mut).any()


def test_with_categorical_values_warn_p(rng):
    srs = pd.Series(list("abcd") * 1_000)

    # only two values from the series can be mutated, so p should be approx. 50%
    df_cat = pd.DataFrame.from_dict({"values": list("ab")})

    mut_categorical = mutator.with_categorical_values(df_cat, value_column="values", rng=rng)

    with pytest.warns(GeckoWarning) as record:
        (srs_mut,) = mut_categorical([srs], 0.8)

    assert len(record) == 1
    assert record[0].message.args[0].startswith("with_categorical_values: desired probability of 0.8 cannot be met")

    srs_mutated_rows = (srs == "a") | (srs == "b")

    assert (srs.loc[srs_mutated_rows] != srs_mut.loc[srs_mutated_rows]).all()
    assert (srs.loc[~srs_mutated_rows] == srs_mut.loc[~srs_mutated_rows]).all()


def test_with_categorical_values_raise_too_few_values(rng):
    df_cat = pd.DataFrame.from_dict({"values": list("a")})

    with pytest.raises(ValueError) as e:
        _ = mutator.with_categorical_values(df_cat, value_column="values", rng=rng)

    assert str(e.value) == "column must contain at least two unique values, has 1"


def test_with_categorical_values_csv(rng, tmp_path):
    srs = pd.Series(list(string.ascii_letters))
    csv_file_path = write_temporary_csv_file(
        tmp_path,
        header=["values"],
        rows=list(string.ascii_letters),
    )

    mut_categorical = mutator.with_categorical_values(csv_file_path, value_column="values", rng=rng)

    (srs_mut,) = mut_categorical([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()


@pytest.mark.parametrize("unit", ["d", "days", "h", "hours", "m", "minutes", "s", "seconds"])
def test_with_datetime_offset(rng, unit):
    srs = pd.Series(pd.date_range("2020-01-01", "2021-01-01", freq="h", inclusive="left"))
    mut_datetime_offset = mutator.with_datetime_offset(5, unit, "%Y-%m-%d %H:%M:%S", rng=rng)
    (srs_mut,) = mut_datetime_offset([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()


@pytest.mark.parametrize("unit", ["d", "days", "h", "hours", "m", "minutes", "s", "seconds"])
def test_with_datetime_offset_partial(rng, unit):
    srs = pd.Series(pd.date_range("2020-01-01", "2021-01-01", freq="h", inclusive="left"))
    mut_datetime_offset = mutator.with_datetime_offset(5, unit, "%Y-%m-%d %H:%M:%S", rng=rng)
    (srs_mut,) = mut_datetime_offset([srs], 0.5)

    assert len(srs) == len(srs_mut)
    assert (srs == srs_mut).any()
    assert (srs != srs_mut).any()


@pytest.mark.parametrize("unit", ["d", "days", "h", "hours", "m", "minutes", "s", "seconds"])
def test_with_datetime_offset_prevent_wraparound(rng, unit):
    srs = pd.Series(["2020-01-01 00:00:00"] * 1_000)
    mut_datetime_offset = mutator.with_datetime_offset(5, unit, "%Y-%m-%d %H:%M:%S", prevent_wraparound=True, rng=rng)

    with pytest.warns(GeckoWarning) as record:
        (srs_mut,) = mut_datetime_offset([srs], 1)

    assert len(record) == 1
    assert record[0].message.args[0].startswith("with_datetime_offset: desired probability of 1 cannot be met")

    assert len(srs) == len(srs_mut)
    assert not (srs != srs_mut).all()
    assert (srs == srs_mut).any()


def test_with_datetime_offset_custom_format(rng):
    srs = pd.Series(pd.date_range("2020-01-01", periods=28, freq="D")).dt.strftime("%d.%m.%Y")
    mut_datetime_offset = mutator.with_datetime_offset(5, "d", "%d.%m.%Y", rng=rng)
    (srs_mut,) = mut_datetime_offset([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert srs_mut.str.fullmatch(r"\d{2}.\d{2}.\d{4}").all()


def test_with_datetime_offset_raise_invalid_delta(rng):
    with pytest.raises(ValueError) as e:
        _ = mutator.with_datetime_offset(0, "d", "%Y-%m-%d", rng=rng)

    assert str(e.value) == "delta must be positive, is 0"


def test_with_phonetic_replacement_table(rng):
    srs = pd.Series(["".join(tpl) for tpl in itertools.permutations("abc")])
    df_phon = pd.DataFrame.from_dict({"source": list("abcbcca"), "target": list("0123456"), "flags": list("^^^$$__")})

    mut_phonetic = mutator.with_phonetic_replacement_table(
        df_phon,
        source_column="source",
        target_column="target",
        flags_column="flags",
        rng=rng,
    )

    (srs_mut,) = mut_phonetic([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert not srs_mut.str.isalpha().all()


def test_with_phonetic_replacement_table_random_values(rng):
    srs = pd.Series(["aaa"] * 1_000)
    df_phon = pd.DataFrame.from_dict({"source": ["a", "a", "a"], "target": ["0", "1", "2"], "flags": ["^", "_", "$"]})

    mut_phonetic = mutator.with_phonetic_replacement_table(
        df_phon,
        source_column="source",
        target_column="target",
        flags_column="flags",
        rng=rng,
    )

    (srs_mut,) = mut_phonetic([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert len(srs_mut.unique()) == 3


def test_with_phonetic_replacement_table_partial(rng):
    srs = pd.Series(["".join(tpl) for tpl in itertools.permutations("abc")])
    df_phon = pd.DataFrame.from_dict({"source": list("abcbcca"), "target": list("0123456"), "flags": list("^^^$$__")})

    mut_phonetic = mutator.with_phonetic_replacement_table(
        df_phon,
        source_column="source",
        target_column="target",
        flags_column="flags",
        rng=rng,
    )

    (srs_mut,) = mut_phonetic([srs], 0.5)

    assert len(srs) == len(srs_mut)
    assert (srs == srs_mut).any()
    assert (srs != srs_mut).any()


def test_with_phonetic_replacement_table_no_flags(rng):
    # no flag defaults to all flags enabled. generate a list of strings where all characters a-z
    # are shuffled.
    srs = pd.Series(
        random_strings(
            charset=string.ascii_lowercase,
            str_len=len(string.ascii_lowercase),
            unique=True,
            rng=rng,
        )
    )

    df_phon = pd.DataFrame.from_dict({"source": ["a"], "target": ["0"], "flags": [""]})

    mut_phonetic = mutator.with_phonetic_replacement_table(
        df_phon,
        source_column="source",
        target_column="target",
        flags_column="flags",
        rng=rng,
    )

    (srs_mut,) = mut_phonetic([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert not srs_mut.str.isalpha().all()


def test_with_phonetic_replacement_table_csv(rng, tmp_path):
    # same as regular test except this one is based on a csv file
    srs = pd.Series(["".join(tpl) for tpl in itertools.permutations("abc")])
    csv_file_path = write_temporary_csv_file(
        tmp_path,
        header=["source", "target", "flags"],
        rows=list(zip("abcbcca", "0123456", "^^^$$__")),
    )

    mut_phonetic = mutator.with_phonetic_replacement_table(
        csv_file_path,
        source_column="source",
        target_column="target",
        flags_column="flags",
        rng=rng,
    )

    (srs_mut,) = mut_phonetic([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert not srs_mut.str.isalpha().all()


def test_with_phonetic_replacement_table_warn_p(rng):
    srs = pd.Series(["abc", "def"] * 100)
    df_phon = pd.DataFrame.from_dict({"source": ["a"], "target": ["0"], "flags": ["^"]})

    mut_phonetic = mutator.with_phonetic_replacement_table(
        df_phon,
        source_column="source",
        target_column="target",
        flags_column="flags",
        rng=rng,
    )

    with pytest.warns(GeckoWarning) as record:
        (srs_mut,) = mut_phonetic([srs], 0.8)

    assert len(record) == 1
    assert (
        record[0]
        .message.args[0]
        .startswith("with_phonetic_replacement_table: desired probability of 0.8 cannot be met")
    )

    srs_mutated_rows = srs.str.startswith("a")

    assert (srs.loc[srs_mutated_rows] != srs_mut.loc[srs_mutated_rows]).all()
    assert (srs.loc[~srs_mutated_rows] == srs_mut.loc[~srs_mutated_rows]).all()


def test_with_phonetic_replacement_table_raise_no_rules(rng):
    with pytest.raises(ValueError) as e:
        _ = mutator.with_phonetic_replacement_table(
            pd.DataFrame.from_dict(
                {
                    "source": [],
                    "target": [],
                    "flags": [],
                }
            ),
            source_column="source",
            target_column="target",
            flags_column="flags",
        )

    assert str(e.value) == "must provide at least one phonetic replacement rule"


def test_with_regex_replacement_table_unnamed_capture_group(rng):
    srs = pd.Series(["abc", "def"] * 100)
    df_table = pd.DataFrame.from_dict({"pattern": ["a(bc)", "d(ef)"], "1": ["1", "2"]})
    mut_regex = mutator.with_regex_replacement_table(df_table, pattern_column="pattern", rng=rng)

    (srs_mut,) = mut_regex([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert not srs_mut.str.isalpha().all()


def test_with_regex_replacement_table_favor_rare_regexes(rng):
    srs = pd.Series(["abc"] * 100 + ["def"] * 50)
    df_table = pd.DataFrame.from_dict({"pattern": ["a(bc)", "d(ef)"], "1": ["1", "2"]})
    mut_regex = mutator.with_regex_replacement_table(df_table, pattern_column="pattern", rng=rng)

    (srs_mut,) = mut_regex([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs_mut == ["a1"] * 100 + ["d2"] * 50).all()


def test_with_regex_replacement_table_unnamed_capture_group_partial(rng):
    srs = pd.Series(["abc", "def"] * 100)

    df_table = pd.DataFrame.from_dict({"pattern": ["a(bc)", "d(ef)"], "1": ["1", "2"]})

    mut_regex = mutator.with_regex_replacement_table(df_table, pattern_column="pattern", rng=rng)

    (srs_mut,) = mut_regex([srs], 0.5)

    assert len(srs) == len(srs_mut)
    assert (srs == srs_mut).any()
    assert (srs != srs_mut).any()


def test_with_regex_replacement_table_named_capture_group(rng):
    srs = pd.Series(["abc", "def"] * 100)

    df_table = pd.DataFrame.from_dict({"pattern": ["a(?P<foo>bc)", "d(?P<foo>ef)"], "foo": ["1", "2"]})

    mut_regex = mutator.with_regex_replacement_table(df_table, pattern_column="pattern", rng=rng)

    (srs_mut,) = mut_regex([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert not srs_mut.str.isalpha().all()


def test_with_regex_replacement_table_flags(rng):
    srs = pd.Series(["abc", "def", "ABC", "DEF"] * 100)

    df_table = pd.DataFrame.from_dict(
        {
            "pattern": ["a(bc)", "d(ef)"],
            "1": ["1", "2"],
            "flags": ["i", "i"],  # ignore case flag
        }
    )

    mut_regex = mutator.with_regex_replacement_table(df_table, pattern_column="pattern", flags_column="flags", rng=rng)

    (srs_mut,) = mut_regex([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert not srs_mut.str.isalpha().all()


def test_with_regex_replacement_table_warn_p(rng):
    srs = pd.Series(["abc", "def"] * 100)

    df_table = pd.DataFrame.from_dict({"pattern": ["a(bc)"], "1": ["1"]})

    mut_regex = mutator.with_regex_replacement_table(df_table, pattern_column="pattern", rng=rng)

    with pytest.warns(GeckoWarning) as record:
        (srs_mut,) = mut_regex([srs], 0.8)

    assert len(record) == 1
    assert (
        record[0].message.args[0].startswith("with_regex_replacement_table: desired probability of 0.8 cannot be met")
    )

    srs_mutated_rows = srs.str.startswith("a")

    assert (srs.loc[srs_mutated_rows] != srs_mut.loc[srs_mutated_rows]).all()
    assert (srs.loc[~srs_mutated_rows] == srs.loc[~srs_mutated_rows]).all()


def test_with_regex_replacement_table_raise_no_rules(rng):
    with pytest.raises(ValueError) as e:
        _ = mutator.with_regex_replacement_table(pd.DataFrame({"pattern": []}), pattern_column="pattern", rng=rng)

    assert str(e.value) == "must provide at least one regex pattern"


def test_with_regex_replacement_table_csv(rng, tmp_path):
    srs = pd.Series(["abc", "def"] * 100)
    csv_path = write_temporary_csv_file(
        tmp_path,
        header=["pattern", "1"],
        rows=[
            ["a(bc)", "1"],
            ["d(ef)", "2"],
        ],
    )

    mut_regex = mutator.with_regex_replacement_table(csv_path, pattern_column="pattern", rng=rng)

    (srs_mut,) = mut_regex([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert not srs_mut.str.isalpha().all()


@pytest.mark.parametrize(
    "pattern",
    [
        [r"^(?P<value>a).", r".(?P<value>b).", r".(?P<value>c)$"],
        [r"^(?P<value>a)\w+", r"\w+(?P<value>b)\w+", r"\w+(?P<value>c)$"],
    ],
)
def test_with_regex_replacement_table_partial(rng, pattern):
    srs = pd.Series(["aaa", "bbb", "ccc"])
    mut_regex = mutator.with_regex_replacement_table(
        pd.DataFrame.from_dict(
            {
                "pattern": pattern,
                "value": ["0", "1", "2"],
            }
        ),
        pattern_column="pattern",
        rng=rng,
    )

    (srs_mut,) = mut_regex([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert (srs_mut == ["0aa", "b1b", "cc2"]).all()


def test_with_regex_replacement_table_random_values(rng):
    srs = pd.Series(["aaa"] * 1_000)
    df_regex_replacement_table = pd.DataFrame.from_dict({"pattern": [".(a).", ".(a).", ".(a)."], "1": ["0", "1", "2"]})

    mut_regex_replacement_table = mutator.with_regex_replacement_table(
        df_regex_replacement_table,
        pattern_column="pattern",
        rng=rng,
    )

    (srs_mut,) = mut_regex_replacement_table([srs], 1)

    assert len(srs) == len(srs_mut)
    assert (srs != srs_mut).all()
    assert len(srs_mut.unique()) == 3


def test_mutate_data_frame(rng):
    lowercase_char_count = len(string.ascii_lowercase)

    def random_unique_lowercase_strings():
        return random_strings(
            n_strings=100_000,
            str_len=lowercase_char_count,
            charset=string.ascii_lowercase,
            unique=True,
            rng=rng,
        )

    df = pd.DataFrame.from_dict(
        {
            "col_1": random_unique_lowercase_strings(),
            "col_2": random_unique_lowercase_strings(),
            "col_3": random_unique_lowercase_strings(),
            "col_4": random_unique_lowercase_strings(),
        }
    )

    df_mut = mutator.mutate_data_frame(
        df,
        [
            # col 1 should have these mutators applied to all rows
            (
                "col_1",
                [
                    mutator.with_delete(rng=rng),
                    mutator.with_uppercase(rng=rng),
                ],
            ),
            # col 2 should have these mutators applied to approx. 50% of all rows
            (
                "col_2",
                [
                    (0.5, mutator.with_insert(charset=string.ascii_uppercase, rng=rng)),
                ],
            ),
            # col 3 and 4 should be permuted
            (("col_3", "col_4"), mutator.with_permute(rng=rng)),
        ],
    )

    # for col_1, ensure that all mutators were applied
    assert (df["col_1"] != df_mut["col_1"]).all()

    assert (df_mut["col_1"].str.len() == lowercase_char_count - 1).all()
    assert df_mut["col_1"].str.upper().all()

    # for col_2, check that the amount of mutated and untouched columns roughly matches
    df_mut_col_2_str_lens = df_mut["col_2"].str.len().value_counts()

    assert (
        abs(df_mut_col_2_str_lens[lowercase_char_count] - df_mut_col_2_str_lens[lowercase_char_count + 1])
        / len(df_mut["col_1"])
        < 0.01
    )

    # for col_3 and col_4, check that the columns were correctly permuted
    assert (df_mut["col_3"] == df["col_4"]).all()
    assert (df_mut["col_4"] == df["col_3"]).all()
