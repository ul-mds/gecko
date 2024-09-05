# Generating data

A generator is a function that takes in a number of records and returns a list of
[Pandas series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html) where each series
represents a data column.
For example, a generator that returns a single column containing numbers from one up to the desired amount of records
could look like this.

```py
import pandas as pd


def generate_numbers(count: int) -> list[pd.Series]:
    numbers = [i + 1 for i in range(count)]
    return [pd.Series(numbers)]
```

Gecko comes with a bunch of built-in generators which are described on this page.
They are exposed in Gecko's `generator` module.

## Available generators

### Frequency tables

One of the most common sources to generate realistic-looking data are frequency tables.
Gecko supports loading frequency tables from CSV files and generating data based off frequencies listed within.

Assume a CSV file containing a list of fruits and their frequencies.
The goal is to generate a series that has a similar distribution of values.

=== "CSV"

    ```csv
    fruit,count
    apple,100
    banana,50
    orange,80
    ```

=== "Table"

    | **Fruit** | **Count** |
    | :-- | --: |
    | Apple | 100 |
    | Banana | 50 |
    | Orange | 80 |

Gecko exposes the function `from_frequency_table` for this purpose.
Point the generator to the CSV file.
Since the columns are named, the value and frequency columns need to be explicitly passed in.

```py
import numpy as np

from gecko import generator

rng = np.random.default_rng(112358)
fruit_generator = generator.from_frequency_table(
    "fruit.csv",
    value_column="fruit",
    freq_column="count",
    rng=rng
)

print(fruit_generator(1000))
# => [["orange", "apple", "apple", "banana", ..., "apple", "apple"]]
```

### Multi-column frequency tables

Oftentimes, frequencies do not depend on a single variable.
For this purpose, Gecko can generate values based off of multiple columns within a CSV file.

Continuing the example from above, assume a frequency table with fruits and their types.

=== "CSV"

    ```csv
    fruit,type,count
    apple,braeburn,30
    apple,elstar,70
    banana,cavendish,40
    banana,plantain,10
    orange,clementine,55
    orange,mandarin,25
    ```

=== "Table"

    | **Fruit** | **Type** | **Count** |
    | :-- | :-- | --: |
    | Apple | Braeburn | 30 |
    | Apple | Elstar | 70 |
    | Banana | Cavendish | 40 |
    | Banana | Plantain | 10 |
    | Orange | Clementine | 55 |
    | Orange | Mandarin | 25 |

These types of frequency tables are handled by the `from_multicolumn_frequency_table` function.
The syntax is similar to that of `from_frequency_table`, except multiple value columns can be passed into it.
This results in a list of series: one for each value column passed into the generator.

```py
import numpy as np

from gecko import generator

rng = np.random.default_rng(14916)
fruit_generator = generator.from_multicolumn_frequency_table(
    "./fruit-types.csv",
    value_columns=["fruit", "type"],
    freq_column="count",
    rng=rng,
)

print(fruit_generator(1000))
# => [["banana", "orange", "apple", "orange", ..., "orange", "banana"],
#     ["cavendish", "mandarin", "elstar", "clementine", ..., "mandarin", "cavendish"]]
```

### Numeric distributions

Gecko provides functions to sample random numbers from uniform and normal distributions.
These are exposed using the `from_uniform_distribution` and `from_normal_distribution` functions.
The numbers are formatted into strings, where the amount of decimal places can be passed to the generators.

The generator for uniform distributions requires an inclusive lower bound and an exclusive upper bound.

```py
import numpy as np

from gecko import generator

rng = np.random.default_rng(2357)
uniform_generator = generator.from_uniform_distribution(
    low=40, high=80, precision=2, rng=rng
)

print(uniform_generator(100))
# => [[47.71, 77.53, 54.93, 50.04, ..., 51.69, 65.63]]
```

The generator for normal distributions requires a mean and a standard deviation.

```py
import numpy as np

from gecko import generator

rng = np.random.default_rng(3731)
normal_generator = generator.from_normal_distribution(
    mean=22, sd=3, precision=2, rng=rng
)

print(normal_generator(100))
# => [[23.77, 17.13, 22.08, 22.07, ..., 21.10, 22.67]]
```

### Date and time information

One of the most commonly collected pieces of identifying information are dates of birth.
More technical sources of dates and times are record creation and update timestamps, as well as other applications of 
tracing data entry.

Gecko provides `from_datetime_range` to generate random timestamps from a uniform distribution.
It can utilize any of [Python's built-in format codes for datetime objects](https://docs.python.org/3/library/datetime.html#format-codes)
to output them to text.

```python
import numpy as np

from gecko import generator


rng = np.random.default_rng(0xcafebabe)
datetime_generator = generator.from_datetime_range(
    start_dt="1920-01-01", 
    end_dt="2020-01-01", 
    dt_format="%d.%m.%Y", 
    unit="D", 
    rng=rng
)

print(datetime_generator(100))
# => [["05.05.1967", "07.06.1923", ..., "09.12.1986", "11.11.1943"]]
```

The "resolution" of the generated strings can be defined by setting the smallest unit of time to alter.
Gecko can generate unique strings down to days (`D`), hours (`h`), minutes (`m`) and seconds (`s`) respectively.
Months are years are currently unsupported since the underlying timespans are nonlinear.

```python
import numpy as np

from gecko import generator


rng = np.random.default_rng(0xdeadbeef)
datetime_generator = generator.from_datetime_range(
    start_dt="1920-01-01", 
    end_dt="2020-01-01", 
    dt_format="%d.%m.%Y %H:%M:%S", 
    unit="m", 
    rng=rng
)

print(datetime_generator(100))
# => [["26.02.1933 17:57:00", "17.12.1954 03:01:00", ..., "15.02.1950 01:29:00", "24.06.1922 23:46:00"]]
```

### Custom generators

Any function that returns a string can be converted into a generator.
Gecko provides `from_function` as a wrapper around such functions.

!!! warning

    You should not use `from_function` if performance matters.
    All built-in generators provided by Gecko are optimized to generate many values at once.
    With `from_function`, new values are generated one by one.

Arguments taken by the wrapped function must be passed to `from_function`.
These arguments are then passed on when values are being generated.
Take the following snippet for example, which generates a random sequence of letters.

```py
import numpy as np
import string

from gecko import generator


def next_letter(
        my_rng: np.random.Generator,
        charset: str = string.ascii_lowercase
):
    return my_rng.choice(list(charset))


rng = np.random.default_rng(11247)

my_generator = generator.from_function(
    next_letter,
    my_rng=rng
)
print(my_generator(100))
# => [["e", "m", "e", "y", ..., "u", "h"]]

my_umlaut_generator = generator.from_function(
    next_letter,
    my_rng=rng,
    charset="äöü"
)
print(my_umlaut_generator(100))
# => [["ü", "ü", "ü", "ä", ..., "ä", "ä"]]
```

An interesting use case is to use Gecko in combination with the
popular [Faker library](https://faker.readthedocs.io/en/master/index.html).
Faker offers many providers for generating synthetic data.
All providers that return strings can be plugged seamlessly into Geckos `from_function` generator.
However, users of Faker are responsible for seeding their own RNG instances to achieve reproducible results.

```py
from faker import Faker
from gecko import generator

fake = Faker("de_DE")
fake.seed_instance(13579)

first_name_generator = generator.from_function(fake.first_name)
age_generator = generator.from_function(
    fake.date_of_birth,
    minimum_age=18,
    maximum_age=80,
)

print(first_name_generator(100))
# => [["Jurij", "Andy", "Gundolf", "Gordana", ..., "Ismet", "Annegrete"]]
print(age_generator(100))
# => [["1969-09-12", "1971-12-15", "1985-03-10", "1949-06-18", ..., "1956-07-26", "1964-09-26"]]
```

## Multiple generators

All generators return one or more series, so it is reasonable to combine them all together into
one [Pandas data frame](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html) for further processing.
Gecko provides the `to_dataframe` function which takes in a list of generators and column names and generates a data
frame based on them.
The following example utilizes most of the generators shown in this guide.

```py
import numpy as np

from gecko import generator

rng = np.random.default_rng(222)

fruit_generator = generator.from_multicolumn_frequency_table(
    "./fruit-types.csv",
    value_columns=["fruit", "type"],
    freq_column="count",
    rng=rng,
)

weight_generator = generator.from_normal_distribution(
    mean=150,
    sd=50,
    precision=1,
    rng=rng,
)

amount_generator = generator.from_uniform_distribution(
    2,
    8,
    precision=0,
    rng=rng,
)


def next_fruit_grade(rand: np.random.Generator) -> str:
    return rand.choice(list("ABC"))


grade_generator = generator.from_function(
    next_fruit_grade,
    rand=rng,
)

df = generator.to_data_frame(
    {
        ("fruit", "type"): fruit_generator,
        "weight_in_grams": weight_generator,
        "amount": amount_generator,
        "grade": grade_generator,
    },
    1_000,
)

print(df)
# => [["fruit", "type", "weight_in_grams", "amount", "grade"],
#       ["apple", "elstar", "162.5", "8", "C"],
#       ["orange", "clementine", "186.8", "5", "A"],
#       ...,
#       ["apple", "elstar", "78.7", "4", "B"]]
```