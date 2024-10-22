Gecko is a Python library for the bulk generation and mutation of realistic personal data.
It is a spiritual successor to the GeCo framework which was initially published by Tran, Vatsalan and Christen.
Gecko reimplements the most promising aspects of the original framework for modern Python with a simplified API, adds
extra features and massively improves performance thanks to NumPy and Pandas.

# Installation

Install with pip:

```bash
pip install gecko-syndata
```

Install with [Poetry](https://python-poetry.org/):

```bash
poetry add gecko-syndata
```

# Basic usage

[Please see the docs for an in-depth guide on how to use the library.](https://ul-mds.github.io/gecko/)

Writing a data generation script with Gecko is usually split into two consecutive steps.
In the first step, data is generated based on information that you provide.
Most commonly, Gecko pulls the information it needs from frequency tables, although other means of generating data
are possible.
Gecko will then output a dataset to your specifications.

In the second step, a copy of this dataset is mutated.
Gecko provides functions which deliberately introduce errors into your dataset.
These errors can take shape in typos, edit errors and other common data sources.
By the end, you will have a generated dataset and a mutated copy thereof.

![Common workflow with Gecko](https://ul-mds.github.io/gecko/img/gecko-workflow.png)

Gecko exposes two modules, `generator` and `mutator`, to help you write data generation scripts.
Both contain built-in functions covering the most common use cases for generating data from frequency information and
mutating data based on common error sources, such as typos, OCR errors and much more.

The following example gives a very brief overview of what a data generation script with Gecko might look like.
It uses frequency tables from the [Gecko data repository](https://github.com/ul-mds/gecko-data) which has been cloned
into a directory next to the script itself.

```python
from pathlib import Path

import numpy as np

from gecko import generator, mutator

# create a RNG with a set seed for reproducible results
rng = np.random.default_rng(727)
# path to the Gecko data repository
gecko_data_dir = Path(__file__).parent / "gecko-data"

# create a data frame with 10,000 rows and a single column called "last_name" 
# which sources its values from the frequency table with the same name
df_generated = generator.to_data_frame(
    [
        ("last_name", generator.from_frequency_table(
            gecko_data_dir / "de_DE" / "last-name.csv",
            value_column="last_name",
            freq_column="count",
            rng=rng,
        )),
    ],
    10_000,
)

# mutate this data frame by randomly deleting characters in 1% of all rows
df_mutated = mutator.mutate_data_frame(
    df_generated,
    [
        ("last_name", (.01, mutator.with_delete(rng))),
    ],
    rng,
)

# export both data frames using Pandas' to_csv function
df_generated.to_csv("german-generated.csv", index_label="id")
df_mutated.to_csv("german-mutated.csv", index_label="id")
```

For a more extensive usage guide, [refer to the docs](https://ul-mds.github.io/gecko/).

# Rationale

The GeCo framework was originally conceived to facilitate the generation and mutation of personal data to validate
record linkage algorithms.
In the field of record linkage, acquiring real-world personal data to test new algorithms on is hard to come by.
Hence, GeCo went for a synthetic approach using statistical models from publicly available data.
GeCo was built for Python 2.7 and has not seen any active development since its last publication in 2013.
The general idea of providing shareable and reproducible Python scripts to generate personal data however still holds a
lot of promise.
This has led to the development of the Gecko library.

A lot of GeCo's weaknesses were rectified with this library.
Vectorized functions from Pandas and NumPy provide significant performance boosts and aid integration into existing
data science applications.
A simplified API allows for a much easier development of custom generators and mutators.
NumPy's random number generation routines instead of Python's built-in `random` module make fine-tuned reproducible
results a breeze.
Gecko therefore seeks to be GeCo's "bigger brother" and aims to provide a much more refined experience to generate
realistic personal data.

# Disclaimer

Gecko is still very much in a "beta" state.
As it stands, it satisfies our internal use cases within the Medical Data Science group, but we also seek wider
adoption.
If you find any issues or improvements with the library, do not hesitate to contact us.

# License

Gecko is released under the MIT License.
