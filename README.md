Gecko is a Python library for the mass generation and corruption of realistic synthetic data.
It is a spiritual successor to the GeCo framework which was initially published by Tran, Vatsalan and Christen.
Gecko reimplements the most promising aspects of the original framework for modern Python with a simplified API, adds
extra features and massively improves performance thanks to Numpy and Pandas.

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

[Please see the docs for an in-depth guide on how to use the library.](https://ul-mds.gitlab.io/record-linkage/gecko/gecko-docs)

Gecko exposes a `generator` and a `corruptor` module.
Both modules contain functions that create generators and corruptors respectively.
Generators are functions that take in a number of values to create and return a list of series, where each series
contains the desired amount of generated values.
Corruptors are functions that take in a list of series and return mutated copies of them.

Create a new generator by importing the `generator` module from the Gecko library.
Pick one of its built-in functions to create a new generator.
To create reproducible results, create
a [new random number generator using Numpy](https://numpy.org/doc/stable/reference/random/generator.html).

```python
import numpy as np

from gecko import generator

rng = np.random.default_rng(727)
generate_numbers = generator.from_uniform_distribution(
    low=10, high=100, precision=2, rng=rng,
)

number_lst = generate_numbers(1000)
srs_numbers = number_lst[0]
print(srs_numbers)
# => [43.55, 84.81, 25.76, ..., 91.06]
```

The above example creates a generator for random numbers drawn from a uniform distribution ranging from 10 to 100.
All numbers are constrained to two decimal places.
The generator is then instructed to generate 1000 values.
Beware that a generator always returns a list of series.
This is because some generators generate multiple series with values that correlate with each other.

Create a new corruptor by importing the `corruptor` module from the Gecko library.
Similar to the `generator` module, pick one of its functions to create a corruptor.

```python
import numpy as np
import pandas as pd

from gecko import corruptor

rng = np.random.default_rng(727)
corrupt_fruits = corruptor.with_delete(rng=rng)

srs_fruits = pd.Series(["apple", "banana", "clementine"])
srs_corrupted = corrupt_fruits([srs_fruits])

print(srs_corrupted)
# => [["aple", "banaa", "cementine"]]
```

The above example creates a corruptor that randomly deletes a single character from each value within a series.
Similar to generators, a corruptor operates on a list of series.

Gecko provides utility functions to use multiple generators at once to generate a data frame, as well as to apply
multiple corruptors on a data frame.

For a more extensive usage guide, [refer to the docs](https://ul-mds.gitlab.io/record-linkage/gecko/gecko-docs).

# Rationale

The GeCo framework was originally conceived to facilitate the generation and corruption of synthetic data to validate
record linkage algorithms.
In the field of record linkage, acquiring real-world personal data to test new algorithms on is hard to come by.
Hence, GeCo went for a synthetic approach using statistical models from publicly available data.
GeCo was built for Python 2.7 and has not seen any active development since its last publication in 2013.
The general idea of providing shareable and reproducible Python scripts to generate synthetic data however still holds a
lot of promise.
This has led to the development of the Gecko library.

A lot of GeCo's weaknesses were rectified with this library.
Vectorized functions from Pandas and Numpy provides significant performance boosts and aid integration into existing
data science applications.
A simplified API allows for a much easier development of custom generators and corruptors.
Numpy's random number generation routines instead of Python's built-in `random` module make fine-tuned reproducible
results a breeze.
Gecko therefore seeks to be GeCo's "bigger brother" and aims to provide a much more refined experience to generate
synthetic but realistic-looking data.

# Disclaimer

Gecko is still very much in a "beta" state.
As it stands, it satisfies our internal use cases within the Medical Data Science group, but we also seek wider
adoption.
If you find any issues or improvements with the library, do not hesitate to contact us.

# License

Gecko is released under the MIT License.
