# Intro to Gecko

Gecko is a Python library for mass generation and corruption of realistic synthetic data.
It aims to be the bigger brother of the GeCo framework which was originally published by Tran et al. in 2013[^1].

Gecko brings a lot of quality-of-life and performance improvements over its predecessor.
It is backed by Numpy and Pandas which allows for easy integration into existing data science applications, as well as massively improved performance by leveraging their vectorized functions wherever possible.

The aim of Gecko is to provide a library which allows for the creation of shareable Python scripts that generate reliable and reproducible synthetic datasets.

## Installation

To get started, install Gecko using your preferred package management tool.

With pip:

```bash
pip install gecko-syndata
```

With Poetry:

```bash
poetry add gecko-syndata
```

## Overview

Gecko consists of two modules: `generator` and `corruptor`.
These modules are responsible for, as their name implies, generating and corrupting data respectively.

<figure markdown>
![Gecko architecture diagram](img/gecko-architecture.png)
<figcaption>Gecko library overview</figcaption>
</figure>

A generator is any function that takes in a number of values to create and returns a list of [series](https://pandas.pydata.org/pandas-docs/stable/reference/series.html), where each series contains the desired amount of generated values.
Generators are expressed by the following type alias.

```python
from typing import Callable
import pandas as pd

# Generator type definition.
Generator = Callable[[int], list[pd.Series]]

# This is what a valid generator looks like.
def my_generator(count: int) -> pd.Series:
    pass
```

A corruptor is any function that takes in a list of series and returns a modified copy of it.
Similar to generators, they are expressed with a simple type alias.

```python
from typing import Callable
import pandas as pd

# Corruptor type definition.
Corruptor = Callable[[list[pd.Series]], list[pd.Series]]

# This is what a valid corruptor looks like.
def my_corruptor(srs_lst: list[pd.Series]) -> list[pd.Series]:
    pass
```

The `generator` and `corruptor` modules provide functions to create generators and corruptors for a range of use cases.
Gecko provides built-in functions to create generators based on frequency tables and numeric distributions, as well as corruptors based on keyboard maps, phonetic errors, edit errors and much more.

You can freely define your own generators and corruptors, or have Gecko create wrappers around existing functions.
There are also helper functions which allow you to chain multiple generators together to create one [data frame](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html) and to apply multiple corruptors to a data frame at once. 

The remainder of this documentation provides a deep dive into the `generator` and `corruptor` modules.
They showcase Gecko's built-in capabilities with plenty of examples.

[^1]: See: Tran, K. N., Vatsalan, D., & Christen, P. (2013, October). GeCo: an online personal data generator and corruptor. In *Proceedings of the 22nd ACM international conference on Information & Knowledge Management* (pp. 2473-2476).