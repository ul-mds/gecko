# Example: German population dataset

In this example, you will create a script that generates and mutates a dataset of German persons.
To get started, create a new directory for your project.
Within it, create an empty `main.py` file.
[Install Gecko](../index.md) and obtain a copy of
the [Gecko data repository](https://github.com/ul-mds/gecko-data).
If you have Git installed, you can simply clone it into your project directory with the following command.

```bash
git clone https://github.com/ul-mds/gecko-data.git
```

Your project directory should look something like this:

```
|- gecko-data/
|- main.py
|- poetry.lock       (only if you use Poetry)
|- pyproject.toml    (only if you use Poetry)
```

The Gecko data repository contains mostly CSV files which have been generated from publicly available datasets.
It is a decent starting point if you just want to explore Gecko's capabilities.

## Setting up the script

To get started, edit your `main.py` and import Numpy, Pandas and Gecko's generator and mutator module.
Numpy is needed to create a random number generator (RNG) with a fixed starting point so that the generated data remains
consistent across script executions.
Most modern code editors should be able to make use of that.

```python
import numpy as np  # (1)!

from gecko import generator, mutator

if __name__ == "__main__":  # (2)!
    rng = np.random.default_rng(727)  # (3)!
```

1. Numpy is imported to make use of its random number generator (RNG). Numpy's RNG functions are extensively used in
   Gecko's built-in generators and mutators.
2. This is standard Python boilerplate code. All code in this block is executed when the script is explicitly run
   using `python main.py`.
3. This line creates a new RNG with a fixed starting point. You can choose any number you want here. Go crazy!

_Tip: Click on the :material-plus-circle: icon in the code snippets to get extra info on what a line of code does._

Next, add two functions to your script.
`generate_data_frame` takes in a number of records to generate and a RNG.
For now this function is empty, but when it's complete, it will return
a [data frame](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html) with the desired amount of rows.
`mutate_data_frame` takes in a data frame and a RNG and returns a modified copy of the data frame that is passed into
it.

Add two lines to your main code block to call the functions that you just created.
Set the amount of records to any number you want.
In this example, the aim is to generate 100k records.

```py
import numpy as np

from gecko import generator, mutator


def generate_data_frame(count, rng):  # (1)!
    pass


def mutate_data_frame(df, rng):  # (2)!
    pass


if __name__ == "__main__":
    rng = np.random.default_rng(727)
    df_original = generate_data_frame(100_000, rng)  # (3)!
    df_mutated = mutate_data_frame(df_original, rng)
```

1. Once complete, this function will generate a data frame with generated data. The amount of rows is set with
   the `count` parameter. Data frames are, in essence, Pandas' representation of two-dimensional tabular data.
2. Once complete, this function will mutate the data frame that is passed into it and return the mutated copy. `df`
   is a common shorthand for "data frame".
3. Gecko works with Pandas data types. If you are familiar with Pandas, you will find working with the results generated
   by Gecko to be easy. A few use cases will be demonstrated later in this example.

## Generating data

Now it's finally time to generate some data.
The Gecko data repository contains a CSV file for the most common last names found in Germany.
It consists of two columns: `last_name` and `count`.
To use it with Gecko, use the `from_frequency_table` function from the `generator` module.

```python
from gecko import generator


def generate_data_frame(count, rng):  # (1)!
    gen_last_name = generator.from_frequency_table(
        "gecko-data/de_DE/last-name.csv",  # (2)!
        value_column="last_name",  # (3)!
        freq_column="count",  # (4)!
        rng=rng,  # (5)!
    )
```

1. For brevity, the rest of the script is excluded from this snippet. Bear in mind that all code you write belongs in
   your `main.py` file.
2. This is the path to the CSV file. It's relative to the location of the `main.py` file.
3. This is the name of the column in the CSV file that contains the values to generate.
4. This is the name of the column in the CSV file that contains the absolute frequency per value.
5. Almost every function in Gecko allows for a RNG to be passed in. Reusing the same RNG in your script is crucial to
   ensure reproducibility. If no RNG is passed in, Gecko will use a new RNG with a random starting point.

Last names aren't everything though.
Next to last names, the Gecko data repository provides a CSV file with given names, as well as a gender code to state
whether a given name is more predominantly assigned to males or females.

This highlights an important aspect about generating realistic data.
Even though data in the real world might seem random, it rarely is.
Names depend on gender at birth, ethnicity, trends in baby names over time and many other social factors.
Gecko provides the tools to generate data according to these complex dependencies.
Since this is an entry-level example, we will keep it as simple as possible.
But if you start to curate your own data sources for use with Gecko, keep in mind that more variables make for a more
realistic dataset.

To generate data from CSV files with multiple interdependent columns, use Gecko's `from_multicolumn_frequency_table`
function.
Its syntax is almost the same as `from_frequency_table`, except it allows for multiple value columns to be specified.

```python
from gecko import generator


def generate_data_frame(count, rng):
    gen_last_name = generator.from_frequency_table(
        "gecko-data/de_DE/last-name.csv",
        value_column="last_name",
        freq_column="count",
        rng=rng,
    )

    gen_given_name_gender = generator.from_multicolumn_frequency_table(
        "gecko-data/de_DE/given-name-gender.csv",
        value_columns=["given_name", "gender"],  # (1)!
        freq_column="count",
        rng=rng,
    )
```

1. Note how this function accepts multiple column names. The generator will use this information to generate multiple
   columns at once.

Gecko can provide generators that create one or two columns at a time, but can it do more than that?
Of course.
There are technically no limits to the amount of interdependent data it can generate ... except the computational and
memory constraints of your machine.
So let's add one more complex generator.

The Gecko data repository has a CSV file with a non-exhaustive list of street names in Germany.
As stated previously, street names are not random just like every other piece of personal information, so the file also
lists the municipalities and post codes where each street can be found.
Again, use `from_multicolumn_frequency_table` and pass in the corresponding column names to generate data from.

```python
from gecko import generator


def generate_data_frame(count, rng):
    gen_last_name = generator.from_frequency_table(
        "gecko-data/de_DE/last-name.csv",
        value_column="last_name",
        freq_column="count",
        rng=rng,
    )

    gen_given_name_gender = generator.from_multicolumn_frequency_table(
        "gecko-data/de_DE/given-name-gender.csv",
        value_columns=["given_name", "gender"],
        freq_column="count",
        rng=rng,
    )

    gen_street_municip_postcode = generator.from_multicolumn_frequency_table(
        "gecko-data/de_DE/street-municipality-postcode.csv",
        value_columns=["street_name", "municipality", "postcode"],  # (1)!
        freq_column="count",
        rng=rng,
    )
```

1. Gecko can work with any amount of value columns, as long as your machine allows for the generation of all that data,
   of course.

To tie it all together, Gecko offers the `to_data_frame` function.
Pass it a list of generators and the column names for each generator and Gecko will generate a data frame according to
your specification.

```python
from gecko import generator


def generate_data_frame(count, rng):
    gen_last_name = generator.from_frequency_table(
        "gecko-data/de_DE/last-name.csv",
        value_column="last_name",
        freq_column="count",
        rng=rng,
    )

    gen_given_name_gender = generator.from_multicolumn_frequency_table(
        "gecko-data/de_DE/given-name-gender.csv",
        value_columns=["given_name", "gender"],
        freq_column="count",
        rng=rng,
    )

    gen_street_municip_postcode = generator.from_multicolumn_frequency_table(
        "gecko-data/de_DE/street-municipality-postcode.csv",
        value_columns=["street_name", "municipality", "postcode"],
        freq_column="count",
        rng=rng,
    )

    return generator.to_data_frame({
        ("given_name", "gender"): gen_given_name_gender,
        "last_name": gen_last_name,
        ("street_name", "municipality", "postcode"): gen_street_municip_postcode,
    }, count)
```

1. The `to_data_frame` function takes two arguments: a list of generators and column names, and the number of records to
   generate.
2. You must provide one or multiple column names for each generator, depending on how many columns a generator creates.
3. If a generator returns only a single column, you can provide a single string. Otherwise, you must provide a list of
   strings.

## Mutating data

Now that we have a data frame with lots of generated data, we can mutate it.
Gecko's `mutator` module exposes the `mutate_data_frame` function which allows you to apply any number of mutators
on the columns of a data frame.

A common source of errors in a dataset are optical character recognition (OCR) errors.
This happens when a physical document is scanned into an image and an OCR software attempts to extract its textual
contents.
The Gecko data repository contains a file with common OCR errors which, in fact, has been sourced straight from the
original GeCo framework which inspired Gecko.

To perform random inline replacements of letters within a word, use the `with_replacement_table` function from
the `mutator` module.
You might notice that this function does not require extra configuration to read the CSV file with the OCR replacements.
If left unconfigured, Gecko will select the first and second column in this case.
This is exactly what we want, so we can leave all other parameters unset.
For this example, apply the mutator to 10% of all values in the `given_name` column.

```python
from gecko import mutator


def mutate_data_frame(df, rng):
    return mutator.mutate_data_frame(df, [  # (1)!
        ("given_name", [
            (0.1, mutator.with_replacement_table(  # (2)!
                "gecko-data/common/ocr.csv",
                rng=rng,
            ))
        ])
    ], rng=rng)  # (3)!
```

1. The `mutate_data_frame` function takes in two arguments: the data frame to mutate and a list. This list
   maps column names to a list of mutators to apply to this column.
2. Each column is assigned a list. This list contains entries that define how mutators should be applied. The syntax
   for each entry is `(probability, mutator)`. So in this case, the replacement table mutator is applied to 10% of
   all values in the `given_name` column. The remaining 90% remain untouched.
3. It is important to supply the RNG to `mutate_data_frame` so that it always selectes the same records for mutation
   every time the script is run.

For columns such as `gender`, the available options are limited.
In this example, it can only take on the values `m` and `f`.
Replacing it with anything else wouldn't make a lot of sense.

Columns with a limited set of permitted values can be mutated using the `with_categorical_values` function.
This ensures that values within this column are only replaced with another valid value.
You can reuse the CSV file containing given names and gender codes for this purpose.
Point the mutator to the `gender` column, and it'll automatically pick out all unique values within it.
For this example, the mutator should modify 2% of all rows.

```python
from gecko import mutator


def mutate_data_frame(df, rng):
    return mutator.mutate_data_frame(df, [
        ("given_name", [
            (0.1, mutator.with_replacement_table(
                "gecko-data/common/ocr.csv",
                rng=rng,
            ))
        ]),
        ("gender", [
            (0.02, mutator.with_categorical_values(  # (1)!
                "gecko-data/de_DE/given-name-gender.csv",
                value_column="gender",
                rng=rng,
            ))
        ])
    ], rng=rng)
```

1. The `with_categorical_values` function allows you to reuse the same files that you used to generate your data. The
   function call is similar to the frequency table functions in the `generator` module.

So far all you've done is apply a single mutator to a column, but Gecko allows you to apply as many mutators as you
want to a column.
Let's suppose that a few entries in the `gender` column are supposed to be missing.
The `with_missing_value` function handles this exact use case.
It replaces values with a representative "missing value", which is an empty string by default.
Extend the list of mutators for the `gender` column by applying the missing value mutator to 5% of all records.

```python
from gecko import mutator


def mutate_data_frame(df, rng):
    return mutator.mutate_data_frame(df, [
        ("given_name", [
            (0.1, mutator.with_replacement_table(
                "gecko-data/common/ocr.csv",
                rng=rng,
            ))
        ]),
        ("gender", [
            (0.02, mutator.with_categorical_values(
                "gecko-data/de_DE/given-name-gender.csv",
                value_column="gender",
                rng=rng,
            )),
            (0.05, mutator.with_missing_value(
                value="",  # (1)!
                strategy="all",  # (2)!
            ))
        ])
    ], rng=rng)
```

1. By default, this mutator uses an empty string as the "missing value". You don't need to add this parameter if you
   are happy with empty strings, but it's good practice nonetheless to be as explicit as possible in case the default
   value changes in the future.
2. By setting the mutator's strategy to `all`, it ensures that all of the randomly selected records will have their
   values replaced with an empty string. The other options, `blank` and `empty`, are explained in
   the [documentation on mutating data](../data-mutation.md#missing-values).

Another common source of errors are typos on a keyboard.
Gecko can read keymaps from the Unicode Common Locale Data Repository (CLDR) and apply typos based on them.
[Download the German CLDR keymap](https://github.com/unicode-org/cldr/blob/release-43/keyboards/windows/de-t-k0-windows.xml)
and place it next to your `main.py` script.

We'll focus on the `postcode` column this time and assume that someone might slip with their finger during data entry
and enter a wrong digit from time to time.
The number row on the German keyboard is surrounded by many keys that don't generate digits.
The `with_cldr_keymap_file` function accounts for that, granted that you pass it a string of allowed characters.

```python
from gecko import mutator


def mutate_data_frame(df, rng):
    return mutator.mutate_data_frame(df, [
        ("given_name", [
            (0.1, mutator.with_replacement_table(
                "gecko-data/common/ocr.csv",
                rng=rng,
            ))
        ]),
        ("gender", [
            (0.02, mutator.with_categorical_values(
                "gecko-data/de_DE/given-name-gender.csv",
                value_column="gender",
                rng=rng,
            )),
            (0.05, mutator.with_missing_value(
                value="",
                strategy="all",
            ))
        ]),
        ("postcode", [
            (0.01, mutator.with_cldr_keymap_file(
                "de-t-k0-windows.xml",  # (1)!
                charset="0123456789",  # (2)!
                rng=rng,
            ))
        ])
    ], rng=rng)
```

1. The `with_cldr_keymap_file` function can read any CLDR
   keymap. [Beware of limitations however](../data-mutation.md#keyboard-typos) since CLDR keymaps are currently
   undergoing a large revision.
2. By constraining the mutator this way, digits on the German keyboard can only be replaced with neighboring digits.

## Putting it all together

It's done!
Well, for now at least.

Gecko provides many more functions for generating and mutating realistic data.
This example is a primer to show you the ropes.
Feel free to extend this basic example with your own generators and mutators.

To wrap things up, export the original and mutated data frames into their own CSV files.
You can
use [Pandas' `to_csv` function](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html)
to accomplish this.
Your final script should look something like this.

```python
import numpy as np

from gecko import generator, mutator


def generate_data_frame(count, rng):
    gen_last_name = generator.from_frequency_table(
        "gecko-data/de_DE/last-name.csv",
        value_column="last_name",
        freq_column="count",
        rng=rng,
    )

    gen_given_name_gender = generator.from_multicolumn_frequency_table(
        "gecko-data/de_DE/given-name-gender.csv",
        value_columns=["given_name", "gender"],
        freq_column="count",
        rng=rng,
    )

    gen_street_municip_postcode = generator.from_multicolumn_frequency_table(
        "gecko-data/de_DE/street-municipality-postcode.csv",
        value_columns=["street_name", "municipality", "postcode"],
        freq_column="count",
        rng=rng,
    )

    return generator.to_data_frame({
        ("given_name", "gender"): gen_given_name_gender,
        "last_name": gen_last_name,
        ("street_name", "municipality", "postcode"): gen_street_municip_postcode,
    }, count)


def mutate_data_frame(df, rng):
    return mutator.mutate_data_frame(df, [
        ("given_name", [
            (0.1, mutator.with_replacement_table(
                "gecko-data/common/ocr.csv",
                rng=rng,
            ))
        ]),
        ("gender", [
            (0.02, mutator.with_categorical_values(
                "gecko-data/de_DE/given-name-gender.csv",
                value_column="gender",
                rng=rng,
            )),
            (0.05, mutator.with_missing_value(
                value="",
                strategy="all",
            ))
        ]),
        ("postcode", [
            (0.01, mutator.with_cldr_keymap_file(
                "de-t-k0-windows.xml",
                charset="0123456789",
                rng=rng,
            ))
        ])
    ], rng=rng)


if __name__ == "__main__":
    rng = np.random.default_rng(727)
    df_original = generate_data_frame(100_000, rng)
    df_mutated = mutate_data_frame(df_original, rng)
    df_original.to_csv("german-original.csv", index_label="id")  # (1)!
    df_mutated.to_csv("german-mutated.csv", index_label="id")
```

1. This highlights one of the main reasons why Gecko works on Pandas data types. It facilitates the inclusion into
   regular data science applications. You can use all your knowledge of Pandas on the results that Gecko provides to
   you.

All that's left to do is to run `python main.py` and examine the fruits of your labor.

## Bonus: Writing your own generator

Gecko comes with a lot of built-in functions, but you are free to write your own generators.
Remember: a generator is a function that takes in a number of records and returns a list
of [Pandas series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html).
As long as your custom function abides by this, Gecko is happy to work with it.

Suppose you want to generate a random date of birth for every person in your synthetic dataset.
Add a new function to your script called `create_date_of_birth_generator`.

```python
import numpy as np
import pandas as pd

from gecko.generator import Generator
from typing import Optional


def create_date_of_birth_generator(
        start_date: str = "1920-01-01",  # (1)!
        end_date: str = "2000-01-01",
        rng: Optional[np.random.Generator] = None,  # (2)!
) -> Generator:
    pass
```

1. If no start or end date is specified, the function will use January 1st, 1920 and January 1st, 2000 as the default
   boundaries.
2. Almost all built-in functions in Gecko allow a custom RNG to be passed in. If no RNG is specified, a new RNG is used.
   It's a good idea to stick to this convention in your own generators.

Some more boilerplate code is required.
First, if `rng` is `None`, then the function should use a new RNG with a random starting point.
Second, your function needs to return a generator, so create a nested function that follows the requirements for it to
be recognized as a generator.

```python
import numpy as np
import pandas as pd

from gecko.generator import Generator
from typing import Optional


def create_date_of_birth_generator(
        start_date: str = "1920-01-01",
        end_date: str = "2000-01-01",
        rng: Optional[np.random.Generator] = None,
) -> Generator:
    if rng is None:  # (1)!
        rng = np.random.default_rng()

    def _generate(count: int) -> list[pd.Series]:  # (2)!
        return []

    return _generate  # (3)!
```

1. If no RNG is supplied to this function, then it will use a new RNG with a random starting
   point. [See the Numpy docs on
   `default_rng` for more information.](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng)
2. This is the actual generator function that will take care of generating random data. It must take in a number and
   return a list of series.
3. The generator is returned by your function. This means whenever you call `create_date_of_birth_generator` in your
   code, you will actually receive the nested `_generate` function.

The way we're going to generate random dates in this example is to take the start date and add a random amount of days
to it.
[Numpy offers functions that simplify working with dates and times.](https://numpy.org/doc/stable/reference/arrays.datetime.html)

First, you need to parse the start and end dates into Numpy's `datetime64` objects.
To get the amount of days between the start and end date, subtract the start date from the end date to obtain
a `timedelta64`.

Converting `timedelta64` into a number of days isn't as trivial though.
To make this work, divide it by a time delta of one day.
This returns a floating point number, which you can then convert into an integer.

```python
import numpy as np
import pandas as pd

from gecko.generator import Generator
from typing import Optional


def create_date_of_birth_generator(
        start_date: str = "1920-01-01",
        end_date: str = "2000-01-01",
        rng: Optional[np.random.Generator] = None,
) -> Generator:
    if rng is None:
        rng = np.random.default_rng()

    def _generate(count: int) -> list[pd.Series]:
        start_dt = np.datetime64(start_date)  # (1)!
        end_dt = np.datetime64(end_date)
        delta = end_dt - start_dt  # (2)!
        days_delta = int(delta / np.timedelta64(1, "D"))  # (3)!

        return []

    return _generate
```

1. Parsing dates in Numpy is as simple as wrapping it into `np.datetime64`, given that the date is provided in ISO 8601
   format.
2. Subtracting `np.datetime64` from one another yields a `np.timedelta64`.
3. By diving the time delta by one day, this line will return the amount of days in the time delta as a floating point
   number. It is then converted into an integer.

Now you can generate random days to add to the start date.
Use [Numpy's RNG
`integers` function](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html)
to quickly generate many numbers at once.
By setting `endpoint` to `True`, you're ensuring that the upper bound is included in the random number generation.

Armed with a random list of numbers, you can add them to the start date and receive a list of random dates ranging from
the specified start to the end date.
Since the list will contain instances of `datetime64`, they will need to be converted into strings.
[`np.char.mod` takes care of that](https://numpy.org/doc/stable/reference/generated/numpy.char.mod.html).

Beware that a generator must return a list of series, so even if you're returning a single series, you must wrap it into
a list.

```python
import numpy as np
import pandas as pd

from gecko.generator import Generator
from typing import Optional


def create_date_of_birth_generator(
        start_date: str = "1920-01-01",
        end_date: str = "2000-01-01",
        rng: Optional[np.random.Generator] = None,
) -> Generator:
    if rng is None:
        rng = np.random.default_rng()

    def _generate(count: int) -> list[pd.Series]:
        start_dt = np.datetime64(start_date)
        end_dt = np.datetime64(end_date)
        delta = end_dt - start_dt
        days_delta = int(delta / np.timedelta64(1, "D"))

        random_days = rng.integers(low=0, high=days_delta, size=count, endpoint=True)  # (1)!
        random_dates = start_dt + random_days  # (2)!
        random_date_strs = np.char.mod("%s", random_dates)  # (3)!

        return [pd.Series(random_date_strs)]  # (4)!

    return _generate
```

1. Numpy's RNG functions are optimized to generate many numbers at once. Gecko uses these functions wherever possible.
   In your code, you should try to use these functions as well as to not affect performance when generating millions of
   records.
2. By adding the list of random days to the starting date, we get a list of random dates.
3. The list of random dates consists of `datetime64` instances, so they need to be converted to strings before being
   returned.
4. Generators must always return a list of series, so even a single series must be wrapped into a list.

And there you go!
Now you have your own generator that you can plug into Gecko with no issues.
Expand the `generate_data_frame` function by adding your own generator.

```python
import numpy as np
import pandas as pd

from gecko.generator import Generator
from typing import Optional

from gecko import generator


def create_date_of_birth_generator(
        start_date: str = "1920-01-01",
        end_date: str = "2000-01-01",
        rng: Optional[np.random.Generator] = None,
) -> Generator:
    if rng is None:
        rng = np.random.default_rng()

    def _generate(count: int) -> list[pd.Series]:
        start_dt = np.datetime64(start_date)
        end_dt = np.datetime64(end_date)
        delta = end_dt - start_dt
        days_delta = int(delta / np.timedelta64(1, "D"))

        random_days = rng.integers(low=0, high=days_delta, size=count, endpoint=True)  # (1)!
        random_dates = start_dt + random_days  # (2)!
        random_date_strs = np.char.mod("%s", random_dates)  # (3)!

        return [pd.Series(random_date_strs)]  # (4)!

    return _generate


def generate_data_frame(count, rng):
    gen_last_name = generator.from_frequency_table(
        "gecko-data/de_DE/last-name.csv",
        value_column="last_name",
        freq_column="count",
        rng=rng,
    )

    gen_given_name_gender = generator.from_multicolumn_frequency_table(
        "gecko-data/de_DE/given-name-gender.csv",
        value_columns=["given_name", "gender"],
        freq_column="count",
        rng=rng,
    )

    gen_street_municip_postcode = generator.from_multicolumn_frequency_table(
        "gecko-data/de_DE/street-municipality-postcode.csv",
        value_columns=["street_name", "municipality", "postcode"],
        freq_column="count",
        rng=rng,
    )

    gen_date_of_birth = create_date_of_birth_generator(rng=rng)  # (1)!

    return generator.to_data_frame({
        ("given_name", "gender"): gen_given_name_gender,
        "last_name": gen_last_name,
        ("street_name", "municipality", "postcode"): gen_street_municip_postcode,
        "date_of_birth": gen_date_of_birth,
    }, count)
```

1. Now you have a reusable generator with a configurable start and end date. As long as a function returns a generator,
   it can be seamlessly used with Gecko.
2. As with all other generators, your custom generator is accepted by `to_data_frame` by simply assigning it a column
   name.