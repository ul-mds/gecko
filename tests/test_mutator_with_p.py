import string
import pandas as pd
import pytest

from gecko.mutator import with_cldr_keymap_file, PNotMetWarning, with_missing_value
from tests.helpers import get_asset_path


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
