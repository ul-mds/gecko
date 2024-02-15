__all__ = [
    "Generator",
    "from_function",
    "from_uniform_distribution",
    "from_normal_distribution",
    "from_frequency_table",
    "from_multicolumn_frequency_table",
    "to_dataframe",
]

from os import PathLike
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from typing_extensions import ParamSpec  # required for 3.9 backport

P = ParamSpec("P")
Generator = Callable[[int], list[pd.Series]]


def from_function(func: Callable[P, str], *args, **kwargs) -> Generator:
    """
    Generate a series from an arbitrary function that returns a single value at a time.
    This generator should be used sparingly as it is not vectorized, meaning values have to be generated one by one.
    Use this generator for testing purposes or if performance is not critical.

    :param func: function that takes in any arguments and returns a single string
    :return: function returning a Pandas series with values generated from the custom function
    """

    def _generate(count: int) -> list[pd.Series]:
        return [pd.Series(data=[func(*args, **kwargs) for _ in np.arange(count)])]

    return _generate


def from_uniform_distribution(
    low: Union[int, float] = 0,
    high: Union[int, float] = 1,
    precision: int = 6,
    rng: Optional[np.random.Generator] = None,
) -> Generator:
    """
    Generate a series of numbers drawn from a uniform distribution within the specified bounds.
    These numbers are formatted into strings.

    :param low: lower (inclusive) bound of the uniform distribution (default: `0`)
    :param high: upper (exclusive) bound of the uniform distribution (default: `1`)
    :param precision: amount of decimal places to round to (default: `6`)
    :param rng: random number generator to use (default: `None`)
    :return: function returning Pandas series of numbers from uniform distribution with specified parameters
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
    rng: Optional[np.random.Generator] = None,
) -> Generator:
    """
    Generate a series of numbers drawn from a normal distribution with the specified parameters.
    These numbers are formatted into strings.

    :param mean: mean of the normal distribution (default: `0`)
    :param sd: standard deviation of the normal distribution (default: `1`)
    :param precision: amount of decimal places to round to (default: `6`)
    :param rng: random number generator to use (default: `None`)
    :return: function returning Pandas series of numbers from normal distribution with specified parameters
    """
    if rng is None:
        rng = np.random.default_rng()

    format_str = f"%.{precision}f"

    def _generate(count: int) -> list[pd.Series]:
        return [pd.Series(np.char.mod(format_str, rng.normal(mean, sd, count)))]

    return _generate


def from_frequency_table(
    csv_file_path: Union[str, PathLike[str]],
    header: bool = False,
    value_column: Union[str, int] = 0,
    freq_column: Union[str, int] = 1,
    encoding: str = "utf-8",
    delimiter: str = ",",
    rng: Optional[np.random.Generator] = None,
) -> Generator:
    """
    Generate a series of values from a CSV file.
    This CSV file must contain at least two columns, one holding values and one holding their absolute frequencies.
    Values are generated using their assigned absolute frequencies.
    Therefore, the values in the resulting series should have a similar distribution compared to the input file.

    :param csv_file_path: CSV file to read from
    :param header: `True` if the file contains a header, `False` otherwise (default: `False`)
    :param value_column: name of the value column if the file contains a header, otherwise the column index (default: `0`)
    :param freq_column: name of the frequency column if the file contains a header, otherwise the column index (default: `1`)
    :param encoding: character encoding of the CSV file (default: `utf-8`)
    :param delimiter: column delimiter (default: `,`)
    :param rng: random number generator to use (default: `None`)
    :return: function returning Pandas series of values with a distribution similar to that of the input file
    """
    if rng is None:
        rng = np.random.default_rng()

    if type(value_column) is not type(freq_column):
        raise ValueError("value and frequency column must both be of the same type")

    # read csv file
    df = pd.read_csv(
        csv_file_path,
        header=0 if header else None,  # header row index (`None` if not present)
        usecols=[value_column, freq_column],
        dtype={freq_column: "int", value_column: "str"},
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
    csv_file_path: Union[str, PathLike[str]],
    header: bool = False,
    value_columns: Union[int, str, list[int], list[str]] = 0,
    freq_column: Union[int, str] = 1,
    encoding: str = "utf-8",
    delimiter: str = ",",
    rng: Optional[np.random.Generator] = None,
) -> Generator:
    """
    Generate a series of values from a CSV file where columns are inter-dependent.
    This CSV file must contain at least two columns, one holding values and one holding their absolute frequencies.
    This corruptor generates a series for each value column that's passed into it.
    Values are generated using their assigned absolute frequencies.
    Therefore, the values in the resulting series should have a similar distribution compared to the input file.

    :param csv_file_path: CSV file to read from
    :param header: `True` if the file contains a header, `False` otherwise (default: `False`)
    :param value_columns: name of the volue column(s) if the file contains a header, otherwise the column index/indices (default: `0`)
    :param freq_column: name of the frequency column(s) if the file contains a header, otherwise the column index/indices (default: `1`)
    :param encoding: character encoding of the CSV file (default: `utf-8`)
    :param delimiter: column delimiter (default: `,`)
    :param rng: random number generator to use (default: `None`)
    :return: function returning Pandas series of values with a distribution similar to that of the input file
    """
    if rng is None:
        rng = np.random.default_rng()

    # if the value columns are a list, then read the type of its entries from the first one
    if isinstance(value_columns, list):
        if len(value_columns) == 0:
            raise ValueError("value column list cannot be empty")

        value_columns_type = type(value_columns[0])
    else:
        value_columns_type = type(value_columns)

    if value_columns_type is not type(freq_column):
        raise ValueError("value and frequency column must both be of the same type")

    # if value_columns is an int or str, wrap it into a list
    value_columns = (
        [value_columns] if not isinstance(value_columns, list) else value_columns
    )

    df = pd.read_csv(
        csv_file_path,
        header=0 if header else None,
        usecols=value_columns + [freq_column],
        dtype={
            freq_column: "int",
            **{value_column: "str" for value_column in value_columns},
        },
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


def to_dataframe(
    column_to_generator_dict: dict[Union[str, tuple[str, ...]], Generator],
    count: int,
):
    """
    Generate a dataframe by using multiple generators at once.
    This function takes a list of generators and the names for each column that a generator will create.

    :param column_to_generator_dict: dict that maps column names to generators
    :param count: number of records to generate
    :return: dataframe with columns generated as specified
    """
    if len(column_to_generator_dict) == 0:
        raise ValueError("generator dict may not be empty")

    if count <= 0:
        raise ValueError(f"amount of rows must be positive, is {count}")

    col_to_srs_dict: dict[str, pd.Series] = {}

    for gen_col_names, gen in column_to_generator_dict.items():
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
