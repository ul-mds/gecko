import csv
from pathlib import Path
from typing import Callable, ParamSpec, Type

import numpy as np
from numpy.random import Generator

P = ParamSpec('P')

CallableGeneratorFunc = Callable[P, str]
GeneratorFunc = Callable[[int], list[list[str]]]


def from_function(
        func: CallableGeneratorFunc,
        *args,
        **kwargs
) -> GeneratorFunc:
    def _generate(count: int) -> list[list[str]]:
        return [[func(*args, **kwargs) for _ in np.arange(count)]]

    return _generate


def _generate_from_uniform_distribution_float(
        rng: Generator,
        low: float,
        high: float
) -> GeneratorFunc:
    def _generate(count: int) -> list[list[str]]:
        return [np.char.mod("%f", rng.uniform(low, high, count))]

    return _generate


def _generate_from_uniform_distribution_int(
        rng: Generator,
        low: int,
        high: int
) -> GeneratorFunc:
    def _generate(count: int) -> list[list[str]]:
        return [np.char.mod("%d", rng.integers(low, high, size=count))]

    return _generate


def from_uniform_distribution(
        rng: Generator | None = None,
        low: float | int = 0,
        high: float | int = 1,
        dtype: Type[int | float] = float
) -> GeneratorFunc:
    if rng is None:
        rng = np.random.default_rng()

    if dtype is float:
        return _generate_from_uniform_distribution_float(rng, low, high)

    if dtype is int:
        return _generate_from_uniform_distribution_int(rng, low, high)

    raise NotImplementedError(f"unexpected data type: {dtype}")


def from_normal_distribution(
        rng: Generator | None = None,
        mean: float = 0,
        sd: float = 1
) -> GeneratorFunc:
    def _generate(count: int) -> list[list[str]]:
        return [np.char.mod("%f", rng.normal(mean, sd, count))]

    return _generate


def from_frequency_table(
        csv_file_path: Path,
        header: bool = False,
        encoding: str = "utf-8",
        delimiter: str = ",",
        rng: Generator | None = None
) -> GeneratorFunc:
    if rng is None:
        rng = np.random.default_rng()

    with csv_file_path.open(mode="r", encoding=encoding, newline="") as f:
        # create csv reader instance
        reader = csv.reader(f, delimiter=delimiter)
        # setup local vars
        value_list: list[str] = []
        abs_freq_list: list[int] = []

        # skip header row
        if header:
            next(reader)

        for line in reader:
            if len(line) != 2:
                raise ValueError("CSV file must contain two columns")

            line_val, line_freq = line[0], int(line[1])

            if line_freq < 0:
                raise ValueError("absolute frequency must not be negative")

            value_list.append(line_val)
            abs_freq_list.append(line_freq)

    # this works because abs_freq_list is broadcast to a numpy array
    # noinspection PyUnresolvedReferences
    rel_freq_list = abs_freq_list / np.sum(abs_freq_list)

    # return type can be treated as a list
    # noinspection PyTypeChecker
    def _generate(count: int) -> list[list[str]]:
        return [rng.choice(value_list, count, p=rel_freq_list)]

    return _generate


def from_multicolumn_frequency_table(
        csv_file_path: Path,
        encoding: str = "utf-8",
        delimiter: str = ",",
        rng: Generator | None = None,
        column_names: str | list[str] | None = None,
        count_column_name: str = "count"
) -> GeneratorFunc:
    if column_names is None:
        raise ValueError("column names must be defined")

    if type(column_names) is list and len(column_names) == 0:
        raise ValueError("list of column names may not be empty")

    if type(column_names) is str:
        column_names = [column_names]

    if rng is None:
        rng = np.random.default_rng()

    with csv_file_path.open(mode="r", encoding=encoding, newline="") as f:
        # create reader instance
        reader = csv.DictReader(f, delimiter=delimiter)

        # sanity check
        for col_name in column_names + [count_column_name]:
            if col_name not in reader.fieldnames:
                raise ValueError(f"column `{col_name}` is not present in the CSV file")

        column_value_list: list[list] = []
        abs_freq_list: list[int] = []

        for row in reader:
            # rows can be indexed using strings. no idea why the type checker complains here.
            # noinspection PyTypeChecker
            row_freq = int(row[count_column_name])

            if row_freq < 0:
                raise ValueError("absolute frequency must not be negative")

            # see above
            # noinspection PyTypeChecker
            row_tuple = list(row[col_name] for col_name in column_names)

            column_value_list.append(row_tuple)
            abs_freq_list.append(row_freq)

    # this works because abs_freq_list is broadcast to a numpy array
    # noinspection PyUnresolvedReferences
    rel_freq_list = abs_freq_list / np.sum(abs_freq_list)

    def _generate(count: int) -> list[list[str]]:
        # choice() can work with lists of tuples
        # noinspection PyTypeChecker
        x = rng.choice(column_value_list, count, p=rel_freq_list)
        # type conversion because zip will return an iterator over tuples, but we need lists instead
        return list(list(t) for t in zip(*x))

    return _generate


def to_csv(
        file_path: Path,
        generators: list[tuple[GeneratorFunc, str | list[str]]],
        count: int,
        encoding: str = "utf-8",
        delimiter: str = ","
):
    if len(generators) == 0:
        raise ValueError("list of generators may not be empty")

    out_column_names: list[str] = []
    out_values: list[list[str]] = []

    for generator, column_names in generators:
        # if a single string is provided, concat by wrapping it into a list
        if isinstance(column_names, str):
            column_names = [column_names]

        out_column_names += column_names
        generator_values = generator(count)

        if len(generator_values) != len(column_names):
            raise ValueError(f"generator returned {len(generator_values)} columns, but requires {len(column_names)} to "
                             f"fill column(s) for: {','.join(column_names)}")

        out_values += generator_values

    with open(file_path, mode="a", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_column_names, delimiter=delimiter)

        # check if we're at the start of the file
        if f.tell() == 0:
            writer.writeheader()

        writer.writerows([
            {
                out_column_names[j]: out_values[j][i] for j in range(len(out_column_names))
            } for i in range(count)
        ])


if __name__ == "__main__":
    p = Path("./myfile.csv")
    rng = np.random.default_rng(727)

    gen_1 = from_frequency_table(Path(__file__).parent.parent / "data" / "surname.csv", rng=rng)
    gen_2 = from_multicolumn_frequency_table(Path(__file__).parent.parent / "data" / "test.csv", rng=rng,
                                             column_names=["col1", "col2"])
    gen_3 = from_uniform_distribution(rng, 10, 100, int)

    to_csv(p, [
        (gen_1, "surname"),
        (gen_2, ["x", "y"]),
        (gen_3, "age")
    ], 100000)
