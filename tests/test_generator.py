import numpy as np
import pandas as pd
import pytest

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


def test_from_frequency_table_no_header(rng):
    generate_tab = generator.from_frequency_table(
        get_asset_path("freq_table_no_header.csv"),
        rng=rng,
    )

    srs = generate_tab(100)[0]

    assert (srs == "foo").any()
    assert (srs == "bar").any()


def test_from_frequency_table_with_header(rng):
    generate_tab = generator.from_frequency_table(
        get_asset_path("freq_table_header.csv"),
        rng=rng,
        value_column="value",
        freq_column="freq",
    )

    srs = generate_tab(100)[0]

    assert (srs == "foo").any()
    assert (srs == "bar").any()


def test_from_frequency_table_tsv(rng):
    generate_tab = generator.from_frequency_table(
        get_asset_path("freq_table_no_header.tsv"), rng=rng, delimiter="\t"
    )

    srs = generate_tab(100)[0]

    assert (srs == "foo").any()
    assert (srs == "bar").any()


def test_from_frequency_table(rng):
    gen_fruits = generator.from_frequency_table(
        get_asset_path("freq-fruits.csv"),
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


def test_from_datetime_range(rng):
    gen_datetime = generator.from_datetime_range(
        "1920-01-01", "2020-01-01", "%d.%m.%Y", "D", rng=rng
    )

    (dt_srs,) = gen_datetime(100)
    assert dt_srs.str.fullmatch(r"\d{2}\.\d{2}\.\d{4}").all()


def test_from_datetime_range_invalid_start_datetime(rng):
    with pytest.raises(ValueError) as e:
        generator.from_datetime_range("foobar", "2020-01-01", "%d.%m.%Y", "D")

    assert str(e.value).startswith("Error parsing datetime string")


def test_from_datetime_range_invalid_end_datetime(rng):
    with pytest.raises(ValueError) as e:
        generator.from_datetime_range("1920-01-01", "foobar", "%d.%m.%Y", "D")

    assert str(e.value).startswith("Error parsing datetime string")


@pytest.mark.parametrize("unit", ["D", "h", "m", "s"])
def test_from_datetime_range_all_units(rng, unit):
    gen_datetime = generator.from_datetime_range(
        "1920-01-01", "2020-01-01", "%d.%m.%Y %H:%M:%S", unit, rng=rng
    )

    (dt_srs,) = gen_datetime(100)

    df_matches = dt_srs.str.extract(
        r"(?P<date>\d{2}.\d{2}.\d{4}) (?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
    )

    assert df_matches["date"].notna().all()

    hour_not_all_zero = unit in ("h", "m", "s")
    minute_not_all_zero = unit in ("m", "s")
    second_not_all_zero = unit == "s"

    assert (not (df_matches["hour"] == "00").all()) == hour_not_all_zero
    assert (not (df_matches["minute"] == "00").all()) == minute_not_all_zero
    assert (not (df_matches["second"] == "00").all()) == second_not_all_zero


def test_from_datetime_range_end_before_start(rng):
    with pytest.raises(ValueError) as e:
        generator.from_datetime_range(
            "2020-01-01", "1920-01-01", "%d.%m.%Y", "D", rng=rng
        )

    assert (
        str(e.value)
        == "start datetime `2020-01-01` is greater than end datetime `1920-01-01`"
    )


def test_to_dataframe_error_empty_list():
    with pytest.raises(ValueError) as e:
        generator.to_data_frame([], 1000)

    assert str(e.value) == "generator list may not be empty"


def test_to_dataframe_error_count_not_positive():
    with pytest.raises(ValueError) as e:
        generator.to_data_frame(
            [("foo", generator.from_uniform_distribution())],
            0,
        )

    assert str(e.value) == "amount of rows must be positive, is 0"


def test_to_dataframe(rng):
    gen_fruit_types = generator.from_multicolumn_frequency_table(
        get_asset_path("freq-fruits-types.csv"),
        value_columns=["fruit", "type"],
        freq_column="count",
        rng=rng,
    )

    gen_numbers = generator.from_uniform_distribution(
        rng=rng,
    )

    df_row_count = 1000
    df = generator.to_data_frame(
        [
            (("fruit", "type"), gen_fruit_types),
            ("num", gen_numbers),
        ],
        df_row_count,
    )

    assert len(df) == df_row_count

    for col in ["fruit", "type", "num"]:
        assert col in df.columns


def test_from_frequency_table_nan(tmp_path, rng):
    freq_file_path = tmp_path / "freq.csv"
    freq_file_path.write_text('value,freq\n"",1\n"foobar",1\n')

    gen_freq_table = generator.from_frequency_table(
        freq_file_path,
        value_column="value",
        freq_column="freq",
        rng=rng,
    )

    (srs,) = gen_freq_table(100)

    assert pd.notna(srs).all()


def test_from_frequency_table_df(rng):
    df = pd.DataFrame.from_dict({"freq": [10, 5], "value": ["foo", "bar"]})

    gen_freq_table = generator.from_frequency_table(
        df, value_column="value", freq_column="freq", rng=rng
    )

    (srs,) = gen_freq_table(100)
    assert pd.notna(srs).all()

    value_counts = srs.value_counts()
    assert value_counts["bar"] < value_counts["foo"]


def test_from_multicolumn_frequency_table_nan(tmp_path, rng):
    freq_file_path = tmp_path / "freq.csv"
    freq_file_path.write_text('value1,value2,freq\n"","bar",1\n"foo","baz",1\n')

    gen_freq_table = generator.from_multicolumn_frequency_table(
        freq_file_path,
        value_columns=["value1", "value2"],
        freq_column="freq",
        rng=rng,
    )

    (srs1, srs2) = gen_freq_table(100)

    assert pd.notna(srs1).all()
    assert pd.notna(srs2).all()


def test_from_multicolumn_frequency_table_df(rng):
    df = pd.DataFrame.from_dict(
        {
            "freq": [10, 5, 20, 10],
            "value1": ["foo", "foo", "bar", "bar"],
            "value2": ["baz", "bat", "baz", "bat"],
        }
    )

    gen_freq_table = generator.from_multicolumn_frequency_table(
        df,
        value_columns=["value1", "value2"],
        freq_column="freq",
        rng=rng,
    )

    (srs_1, srs_2) = gen_freq_table(100)
    assert pd.notna(srs_1).all()
    assert pd.notna(srs_2).all()

    counts_1 = srs_1.value_counts()
    assert counts_1["bar"] > counts_1["foo"]

    counts_2 = srs_2.value_counts()
    assert counts_2["baz"] > counts_2["bat"]


def test_from_group_single_column_same_weight(rng):
    gen_a = generator.from_function(lambda: "a")
    gen_b = generator.from_function(lambda: "b")

    gen_group = generator.from_group([gen_a, gen_b], rng=rng)
    count = 100_000

    (srs,) = gen_group(count)
    assert len(srs) == count

    # check that the generated values are roughly equally distributed (<0.01%)
    df_value_counts = srs.value_counts()
    assert abs(df_value_counts["a"] - df_value_counts["b"]) / count < 0.0001


def test_from_group_single_column_different_weight(rng):
    gen_a = generator.from_function(lambda: "a")
    gen_b = generator.from_function(lambda: "b")

    count = 100_000
    gen_group = generator.from_group(
        [
            (0.25, gen_a),
            (0.75, gen_b),
        ],
        rng=rng,
    )

    (srs,) = gen_group(count)
    assert len(srs) == count

    # check that the difference in relative frequency (50%) is present
    df_value_counts = srs.value_counts()
    assert (
        abs(0.5 - abs((df_value_counts["a"] - df_value_counts["b"]) / count)) < 0.0001
    )


def test_from_group_multiple_column_same_weight(rng):
    gen_a = lambda c: [pd.Series(["a1"] * c), pd.Series(["a2"] * c)]
    gen_b = lambda c: [pd.Series(["b1"] * c), pd.Series(["b2"] * c)]

    count = 100_000
    gen_group = generator.from_group([gen_a, gen_b], rng=rng)

    (srs_1, srs_2) = gen_group(count)
    assert len(srs_1) == len(srs_2) == count

    df_value_counts_1 = srs_1.value_counts()
    assert abs(df_value_counts_1["a1"] - df_value_counts_1["b1"]) / count < 0.0001

    df_value_counts_2 = srs_2.value_counts()
    assert abs(df_value_counts_2["a2"] - df_value_counts_2["b2"]) / count < 0.0001


def test_from_group_multiple_column_different_weight(rng):
    gen_a = lambda c: [pd.Series(["a1"] * c), pd.Series(["a2"] * c)]
    gen_b = lambda c: [pd.Series(["b1"] * c), pd.Series(["b2"] * c)]

    count = 100_000
    gen_group = generator.from_group(
        [
            (0.25, gen_a),
            (0.75, gen_b),
        ],
        rng=rng,
    )

    (srs_1, srs_2) = gen_group(count)
    assert len(srs_1) == len(srs_2) == count

    # check that the difference in relative frequency (50%) is present
    df_value_counts_1 = srs_1.value_counts()
    assert (
        abs(0.5 - abs((df_value_counts_1["a1"] - df_value_counts_1["b1"]) / count))
        < 0.0001
    )

    df_value_counts_2 = srs_2.value_counts()
    assert (
        abs(0.5 - abs((df_value_counts_2["a2"] - df_value_counts_2["b2"]) / count))
        < 0.0001
    )


def test_from_group_raise_different_column_counts(rng):
    gen_a = generator.from_function(lambda: "a")
    gen_b = lambda c: [pd.Series(["b1"] * c), pd.Series(["b2"] * c)]

    with pytest.raises(ValueError) as e:
        gen = generator.from_group([gen_a, gen_b], rng=rng)
        gen(100_000)

    assert str(e.value) == "generators returned different amounts of columns: got 1, 2"


def test_from_group_raise_p_sum_not_1(rng):
    gen_a = generator.from_function(lambda: "a")
    gen_b = generator.from_function(lambda: "b")

    with pytest.raises(ValueError) as e:
        generator.from_group([(0.2, gen_a), (0.3, gen_b)], rng=rng)

    assert str(e.value) == "sum of weights must be 1, is 0.5"


def test_from_group_raise_row_count(rng):
    gen_a = generator.from_function(lambda: "a")
    gen_b = generator.from_function(lambda: "b")
    gen_c = generator.from_function(lambda: "c")

    with pytest.raises(ValueError) as e:
        gen = generator.from_group([gen_a, gen_b, gen_c], rng=rng)
        gen(100_000)

    assert str(e.value) == (
        "sum of values per generator does not equal amount of desired rows: expected 100000, is 99999 - "
        "this is likely due to rounding errors and can be compensated for by adjusting "
        "`max_rounding_adjustment`"
    )


def test_from_group_rounding_adjustment(rng):
    gen_a = generator.from_function(lambda: "a")
    gen_b = generator.from_function(lambda: "b")
    gen_c = generator.from_function(lambda: "c")

    gen = generator.from_group(
        [gen_a, gen_b, gen_c], rng=rng, max_rounding_adjustment=1
    )

    # this should work without error
    _ = gen(100_000)
