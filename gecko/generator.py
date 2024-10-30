"""
The generator module provides generator functions for generating realistic data.
These generators wrap around common data sources such as frequency tables and numeric distributions.
"""

__all__ = [
    "Generator",
    "from_function",
    "from_uniform_distribution",
    "from_normal_distribution",
    "from_frequency_table",
    "from_multicolumn_frequency_table",
    "from_datetime_range",
    "from_group",
    "to_data_frame",
]

import typing as _t
from os import PathLike

import numpy as np
import pandas as pd
import typing_extensions as _te

_P = _te.ParamSpec("_P")
Generator = _t.Callable[[int], list[pd.Series]]


def from_function(
    func: _t.Callable[_P, str], *args: object, **kwargs: object
) -> Generator:
    """
    Generate data from an arbitrary function that returns a single value at a time.

    Notes:
        This function should be used sparingly since it is not vectorized.
        Only use it for testing purposes or if performance is not important.

    Args:
        func: function to invoke to generate data from
        *args: positional arguments to pass to `func`
        **kwargs: keyword arguments to pass to `func`

    Returns:
        function returning list with strings generated from custom function
    """

    def _generate(count: int) -> list[pd.Series]:
        return [pd.Series(data=[func(*args, **kwargs) for _ in np.arange(count)])]

    return _generate


def from_uniform_distribution(
    low: _t.Union[int, float] = 0,
    high: _t.Union[int, float] = 1,
    precision: int = 6,
    rng: _t.Optional[np.random.Generator] = None,
) -> Generator:
    """
    Generate data from a uniform distribution.

    Args:
        low: lower limit of uniform distribution (inclusive)
        high: upper limit of uniform distribution (exclusive)
        precision: decimal precision of the numbers generated from the uniform distribution
        rng: random number generator to use

    Returns:
        function returning list with numbers drawn from a uniform distribution formatted as strings
    """
    if rng is None:
        rng = np.random.default_rng()

    format_str = f"%.{precision}f"

    def _generate(count: int) -> list[pd.Series]:
        return [pd.Series(np.char.mod(format_str, rng.uniform(low, high, count)))]

    return _generate


def from_normal_distribution(
    mean: float = 0,
    sd: float = 1,
    precision: int = 6,
    rng: _t.Optional[np.random.Generator] = None,
) -> Generator:
    """
    Generate data from a normal distribution.

    Args:
        mean: mean of the normal distribution
        sd: standard deviation of the normal distribution
        precision: decimal precision of the numbers generated from the normal distribution
        rng: random number generator to use

    Returns:
        function returning list with numbers drawn from a normal distribution formatted as strings
    """
    if rng is None:
        rng = np.random.default_rng()

    format_str = f"%.{precision}f"

    def _generate(count: int) -> list[pd.Series]:
        return [pd.Series(np.char.mod(format_str, rng.normal(mean, sd, count)))]

    return _generate


def from_frequency_table(
    data_source: _t.Union[str, PathLike[str], pd.DataFrame],
    value_column: _t.Union[str, int] = 0,
    freq_column: _t.Union[str, int] = 1,
    encoding: str = "utf-8",
    delimiter: str = ",",
    rng: _t.Optional[np.random.Generator] = None,
) -> Generator:
    """
    Generate data from a frequency table.
    The frequency table must be provided in CSV format and contain at least two columns: one containing values to
    generate and one containing their assigned absolute frequencies.
    Values generated by this function will have a distribution similar to the frequencies listed in the input file.
    If the value and frequency column are provided as strings, then it is automatically assumed that the CSV file
    has a header row.

    Args:
        data_source: path to CSV file or data frame to use as frequency table
        value_column: name or index of the value column
        freq_column: name or index of the frequency column
        encoding: character encoding of the CSV file
        delimiter: column delimiter of the CSV file
        rng: random number generator to use

    Returns:
        function returning list with single series containing values generated from the input file
    """
    if rng is None:
        rng = np.random.default_rng()

    if type(value_column) is not type(freq_column):
        raise ValueError("value and frequency columns must be of the same type")

    # skip check for value_column bc they are both of the same type already
    if not isinstance(freq_column, (str, int)):
        raise ValueError(
            "value and frequency columns must be either a string or an integer"
        )

    if isinstance(data_source, pd.DataFrame):
        df = data_source
    else:
        header = isinstance(freq_column, str)

        # read csv file
        df = pd.read_csv(
            data_source,
            header=0 if header else None,  # header row index (`None` if not present)
            usecols=[value_column, freq_column],
            dtype={freq_column: "int", value_column: "str"},
            keep_default_na=False,
            sep=delimiter,
            encoding=encoding,
        )

    # convert absolute to relative frequencies
    srs_value = df[value_column]
    srs_prob = df[freq_column] / df[freq_column].sum()

    def _generate(count: int) -> list[pd.Series]:
        return [pd.Series(rng.choice(srs_value, count, p=srs_prob))]

    return _generate


def from_multicolumn_frequency_table(
    data_source: _t.Union[str, PathLike[str], pd.DataFrame],
    value_columns: _t.Union[int, str, list[int], list[str]] = 0,
    freq_column: _t.Union[int, str] = 1,
    encoding: str = "utf-8",
    delimiter: str = ",",
    rng: _t.Optional[np.random.Generator] = None,
) -> Generator:
    """
    Generate data from a frequency table with multiple interdependent columns..
    The frequency table must be provided in CSV format and contain at least two columns: one containing values to
    generate and one containing their assigned absolute frequencies.
    Values generated by this function will have a distribution similar to the frequencies listed in the input file.
    If the values and frequency column are provided as strings, then it is automatically assumed that the CSV file
    has a header row.

    Args:
        data_source: path to CSV file or data frame to use as frequency table
        value_columns: names or indices of the value columns
        freq_column: name or index of the frequency column
        encoding: character encoding of the CSV file
        delimiter: column delimiter of the CSV file
        rng: random number generator to use

    Returns:
        function returning list with as many series as there are value columns specified containing values generated from the input file
    """
    if rng is None:
        rng = np.random.default_rng()

    # coalesce into list
    if not isinstance(value_columns, list):
        value_columns = [value_columns]

    # check that list is not empty
    if len(value_columns) == 0:
        raise ValueError("value column list cannot be empty")

    # peek at type of first value column
    if type(value_columns[0]) is not type(freq_column):
        raise ValueError("value and frequency columns must be of the same type")

    # skip check for value_columns bc they are both of the same type already
    if not isinstance(freq_column, str) and not isinstance(freq_column, int):
        raise ValueError(
            "value and frequency column must be either a string or an integer"
        )

    # now check that all other entries in the value column are of the same type as the first entry
    # (which has been validated already)
    for i in range(1, len(value_columns)):
        if not isinstance(value_columns[i], type(value_columns[0])):
            raise ValueError(
                "value and frequency column must be either a string or an integer"
            )

    if isinstance(data_source, pd.DataFrame):
        df = data_source
    else:
        header = isinstance(freq_column, str)

        df = pd.read_csv(
            data_source,
            header=0 if header else None,
            usecols=value_columns + [freq_column],
            dtype={
                freq_column: "int",
                **{value_column: "str" for value_column in value_columns},
            },
            keep_default_na=False,
            sep=delimiter,
            encoding=encoding,
        )

    # sum of absolute frequencies
    freq_total = df[freq_column].sum()
    # new series to track the relative frequencies
    value_tuple_list = list(zip(*[list(df[c]) for c in value_columns]))
    rel_freq_list = list(df[freq_column] / freq_total)

    # noinspection PyTypeChecker
    def _generate(count: int) -> list[pd.Series]:
        x = rng.choice(value_tuple_list, count, p=rel_freq_list)
        return [pd.Series(list(t)) for t in zip(*x)]  # dark magic

    return _generate


def from_datetime_range(
    start_dt: _t.Union[str, np.datetime64],
    end_dt: _t.Union[str, np.datetime64],
    dt_format: str,
    unit: _t.Literal["D", "h", "m", "s"],
    rng: _t.Optional[np.random.Generator] = None,
) -> Generator:
    """
    Generate data from a range of dates and times.
    The start and end datetime must be provided either as a ISO 8601 datetime string or a NumPy datetime object.
    The output format must include the same format codes as specified in the `datetime` Python module for the
    `strftime` function.
    The unit specifies the smallest unit of time that may change when generating random dates and times.
    For example if `D` is specified, generated dates will only differ in their days, months and years, leaving hours,
    minutes and seconds unaffected.
    The same applies for `h`, `m` and `s` for hours, minutes and seconds respectively.

    Args:
        start_dt: datetime string or object for start of range
        end_dt: datetime string or object for end of range
        dt_format: output format for generated datetimes
        unit: smallest unit of time that may change when generating random dates and times
        rng: random number generator to use

    Returns:
        function returning list of random datetime strings within the specified range
    """
    if isinstance(start_dt, str):
        start_dt = np.datetime64(start_dt)

    if isinstance(end_dt, str):
        end_dt = np.datetime64(end_dt)

    if start_dt >= end_dt:
        raise ValueError(
            f"start datetime `{start_dt}` is greater than end datetime `{end_dt}`"
        )

    if rng is None:
        rng = np.random.default_rng()

    def _generate(count: int) -> list[pd.Series]:
        delta_td = end_dt - start_dt
        delta_amt = int(delta_td / np.timedelta64(1, unit))
        random_vals = rng.integers(low=0, high=delta_amt, size=count, endpoint=True)
        random_dts = start_dt + random_vals.astype(f"timedelta64[{unit}]")
        dt_srs = pd.Series(random_dts)

        return [dt_srs.dt.strftime(dt_format)]

    return _generate


_WeightedGenerator = tuple[_t.Union[int, float], Generator]


def _is_weighted_generator(x: object) -> _te.TypeGuard[_WeightedGenerator]:
    return (
        isinstance(x, tuple)
        and len(x) == 2
        and isinstance(x[0], (float, int))
        and callable(x[1])
    )


def from_group(
    generator_lst: _t.Union[list[Generator], list[_WeightedGenerator]],
    rng: _t.Optional[np.random.Generator] = None,
) -> Generator:
    if all(callable(g) for g in generator_lst):
        p_per_generator = 1 / len(generator_lst)
        generator_lst = [(p_per_generator, g) for g in generator_lst]

    if not all(_is_weighted_generator(g) for g in generator_lst):
        raise ValueError(
            "invalid argument, must be a list of generators or weighted generators"
        )

    p_sum = sum(g[0] for g in generator_lst)

    if p_sum != 1:
        raise ValueError(f"sum of weights must be 1, is {p_sum}")

    if rng is None:
        rng = np.random.default_rng()

    def _generate(count: int) -> list[pd.Series]:
        p_vals = tuple(g[0] for g in generator_lst)  # get percentage for each generator
        count_per_generator = tuple(
            count * p for p in p_vals
        )  # get absolute counts for each generator
        count_sum = sum(count_per_generator)

        if count_sum != count:
            raise ValueError(
                f"sum of values per generator does not equal amount of desired rows: expected {count}, "
                f"is {count_sum} - this is likely due to rounding errors and cannot be compensated "
                f"for automatically"
            )

        generated_series_lsts: list[list[pd.Series]] = []

        for i, weighted_generator in enumerate(generator_lst):
            _, gen = (
                weighted_generator  # drop first argument since we won't be needing the p for this generator
            )
            generated_series_lsts.append(gen(count_per_generator[i]))

        column_counts = set(len(srs_lst) for srs_lst in generated_series_lsts)

        if len(column_counts) != 1:
            raise ValueError(
                f"generators returned different amounts of columns: "
                f"got {', '.join(str(c) for c in column_counts)}"
            )

        column_count = column_counts.pop()  # get column count

        srs_lst_out = [
            pd.concat(
                [srs_lst[i] for srs_lst in generated_series_lsts],
                axis=0,
                ignore_index=True,
            )
            for i in range(column_count)
        ]

        # reindex randomly
        rand_idx = np.arange(0, count)
        rng.shuffle(rand_idx)

        return [srs.iloc[rand_idx].reset_index(drop=True) for srs in srs_lst_out]

    return _generate


_GeneratorSpec = list[tuple[_t.Union[str, tuple[str, ...]], Generator]]


def to_data_frame(
    generator_lst: _GeneratorSpec,
    count: int,
) -> pd.DataFrame:
    """
    Generate data frame by using multiple generators at once.
    Column names must be mapped to their respective generators.
    A generator can be assigned to one or multiple column names, but it must always match the amount of series
    that the generator returns.

    Args:
        generator_lst: list of column names to generators
        count: amount of records to generate

    Returns:
        data frame with columns and rows generated as specified
    """
    if len(generator_lst) == 0:
        raise ValueError("generator list may not be empty")

    if count <= 0:
        raise ValueError(f"amount of rows must be positive, is {count}")

    col_to_srs_dict: dict[str, pd.Series] = {}

    for col_to_gen_def in generator_lst:
        gen_col_names, gen = col_to_gen_def

        # if a single string is provided, concat by wrapping it into a list
        if isinstance(gen_col_names, str):
            gen_col_names = (gen_col_names,)

        # generate values
        gen_col_values = gen(count)

        # check that the generator returned as many columns as expected
        if len(gen_col_values) != len(gen_col_names):
            raise ValueError(
                f"generator returned {len(gen_col_values)} columns, but requires {len(gen_col_names)} to "
                f"fill column(s) for: {','.join(gen_col_names)}"
            )

        # assign name to series
        for i in range(len(gen_col_values)):
            col_to_srs_dict[gen_col_names[i]] = gen_col_values[i]

    # finally create df from the list of named series
    return pd.DataFrame(data=col_to_srs_dict)
