# Mutating data

A mutator is a function that takes in a list
of [Pandas series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html) where each series
represents a column and returns a mutated copy of it.
For example, a mutator that converts all strings in a series into uppercase could look like this.

```py
import pandas as pd


def mutate_to_all_uppercase(srs_lst: list[pd.Series]) -> list[pd.Series]:
    return [srs.str.upper() for srs in srs_lst]
```

Gecko comes with a bunch of built-in mutators which are described on this page.
They are exposed in Gecko's `mutator` module.

## Available mutators

### Keyboard typos

One of the most common sources for typos are adjacent keys on a keyboard.
Gecko supports loading of keyboard layouts and applying typos based on them.
Currently, keyboard layouts must be provided as an XML file from
the [Unicode CLDR repository](https://github.com/unicode-org/cldr).
Gecko parses these files and determines all neighboring keys of each key, as well as their variants with and without
++shift++ pressed.

!!! warning

    As of Unicode CLDR keyboard specification is under a major redesign as of release 44.
    Support will be added as soon as the specification is finalized.
    For now, please retrieve CLDR keyboard files from a release tagged 43 or earlier.
    The examples in this documentation use files from the CLDR release 43.

[Download the German keyboard layout from the CLDR repository](https://github.com/unicode-org/cldr/blob/release-43/keyboards/windows/de-t-k0-windows.xml).
The corresponding mutator is called `with_cldr_keymap_file`.
Point the mutator to the file you just downloaded.
In the following example, one character in each word is substituted by another neighboring character on the German
keyboard.

```py
import pandas as pd
import numpy as np

from gecko import mutator

rng = np.random.default_rng(3141)
kb_mutator = mutator.with_cldr_keymap_file(
    "./de-t-k0-windows.xml",
    rng=rng
)
srs = pd.Series(["apple", "banana", "clementine"])
print(kb_mutator([srs]))
# => [["spple", "banany", "cldmentine"]]
```

By default, this mutator considers all possible neighboring keys for each key.
If you want to constrain typos to a certain set of characters, you can pass an optional string of characters to this
mutator.
One such example is to limit the mutator to digits when manipulating a series of numbers that are broken up by
non-digit characters.
The following snippet avoids the substitution of hyphens by specifying that only digits may be manipulated.

```py
import pandas as pd
import numpy as np
import string

from gecko import mutator

rng = np.random.default_rng(2718)
kb_mutator = mutator.with_cldr_keymap_file(
    "./de-t-k0-windows.xml",
    charset=string.digits,
    rng=rng
)
srs = pd.Series(["123-456-789", "727-727-727", "294-753-618"])
print(kb_mutator([srs]))
# => [["122-456-789", "827-727-727", "294-753-628"]]
```

### Phonetic errors

One of the most challenging error sources to model are phonetic errors.
These are words that sound the same but are written differently.

In German, for example, "ß" can almost always be replaced with "ss" and still have the word that it's in sound the same.
Whether one writes "Straße" or "Strasse" does not matter as far as pronunciation is concerned.
The same holds for "dt" and "tt" at the end of a word, since both reflect a hard "t" sound.
One can derive rules from similarly sounding character sequences.

Gecko offers a method for modelling these rules and introducing phonetic errors based on them.
A phonetic rule in Gecko consists of a source pattern ("ß", "dt"), a target pattern ("ss", "tt") and positional flags.
The flags determine whether this rule applies at the start (`^`), in the middle (`_`) or the end (`$`) of a word.
These flags can be freely combined.
The absence of a positional flag implies that a rule can be applied anywhere in a string.
Taking the example from above, a suitable rule table could look like this.

=== "CSV"

    ```csv
    source,target,flags
    ß,ss,
    dt,tt,$
    ```

=== "Table"

    | **Source** | **Target** | **Flags** |
    | :-- | :-- | --: |
    | ß | ss | |
    | dt | tt | $ |

Gecko exposes the `with_phonetic_replacement_table` function to handle these types of tables.
The call signature is similar to that of `with_replacement_table`.

```py
import numpy as np
import pandas as pd

from gecko import mutator

rng = np.random.default_rng(8844167)

phonetic_mutator = mutator.with_phonetic_replacement_table(
    "./phonetic-rules-de.csv",
    source_column="source",
    target_column="target",
    flags_column="flags",
    rng=rng,
)

srs = pd.Series(["straße", "stadt", "schießen"])
print(phonetic_mutator([srs]))
# => [["strasse", "statt", "schiessen"]]
```

### Missing values

A textual representation of a "missing value" is sometimes used to clearly indicate that a blank or an empty value is to
be interpreted as a missing piece of information.
In datasets sourced from large databases, this "missing value" might consist of characters that do not adhere to a table
or column schema.
A simple example would be `###_MISSING_###` in place of a person's date of birth, since it does not conform to any
common date format and consists entirely of letters and special characters.

Gecko provides the function `with_missing_value` which replaces certain values within a series with a custom "missing
value".
The mutator replaces either empty, blank or all strings within a series depending on the defined strategy.
This is best explained by a few examples.

Gecko considers strings to be "empty" when their length is zero.
Strings with whitespaces will be left as-is.

```py
import pandas as pd

from gecko import mutator

missing_mutator = mutator.with_missing_value(
    "###_MISSING_###",
    strategy="empty"
)
srs = pd.Series(["apple", "   ", ""])
print(missing_mutator([srs]))
# => [["apple", "   ", "###_MISSING_###"]]
```

Gecko considers strings to be "blank" when their length is zero after trimming all leading and trailing whitespaces.
This is the default behavior of this mutator.

```py
import pandas as pd

from gecko import mutator

missing_mutator = mutator.with_missing_value(
    "###_MISSING_###",
    strategy="blank"
)
srs = pd.Series(["apple", "   ", ""])
print(missing_mutator([srs]))
# => [["apple", "###_MISSING_###", "###_MISSING_###"]]
```

The "nuclear" option is to replace all strings within a series with the "missing value".

```py
import pandas as pd

from gecko import mutator

missing_mutator = mutator.with_missing_value(
    "###_MISSING_###",
    strategy="all"
)
srs = pd.Series(["apple", "   ", ""])
print(missing_mutator([srs]))
# => [["###_MISSING_###", "###_MISSING_###", "###_MISSING_###"]]
```

### Edit errors

Edit errors are caused by a set of operations on single characters within a word.
There are commonly four operations that can induce these types of errors: insertion and deletion of a single character,
substitution of a character with a different one, and transposition of two adjacent characters.

Gecko provides mutators for each of these operations.
For insertions and substitutions, it is possible to define a set of characters to choose from.

```python
import string

import numpy as np
import pandas as pd

from gecko import mutator

rng = np.random.default_rng(8080)
srs = pd.Series(["apple", "banana", "clementine"])

insert_mutator = mutator.with_insert(charset=string.ascii_letters, rng=rng)
print(insert_mutator([srs]))
# => [["aVpple", "banaFna", "clemenMtine"]]

delete_mutator = mutator.with_delete(rng=rng)
print(delete_mutator([srs]))
# => [["aple", "bnana", "clementin"]]

substitute_mutator = mutator.with_substitute(charset=string.digits, rng=rng)
print(substitute_mutator([srs]))
# => [["appl9", "ba4ana", "clementi9e"]]

transpose_mutator = mutator.with_transpose(rng)
print(transpose_mutator([srs]))
# => [["paple", "baanna", "clemenitne"]]
```

Gecko also provides a more general edit mutator which wraps around the insertion, deletion, substitution and
transposition mutator.
It is then possible to assign probabilities for each operation.
By default, all operations are equally likely to be performed.

```python
import numpy as np
import pandas as pd

from gecko import mutator

rng = np.random.default_rng(8443)
srs = pd.Series(["apple", "banana", "clementine", "durian", "eggplant", "fig", "grape", "honeydew"])

edit_mutator_1 = mutator.with_edit(rng=rng)
print(edit_mutator_1([srs]))
# => [["aple", "banan", "clementinb", "duiran", "eAgplant", "Nig", "grapce", "hoKeydew"]]

edit_mutator_2 = mutator.with_edit(
    p_insert=0.1,
    p_delete=0.2,
    p_substitute=0.3,
    p_transpose=0.4,
    rng=rng,
)
print(edit_mutator_2([srs]))
# => [["aplpe", "anana", "lementine", "duriRan", "geggplant", "fg", "rgape", "honedyew"]]
```

### Categorical errors

Sometimes an attribute can only take on a set number of values.
For example, if you have a "gender" column in your dataset and it can only take on `m` for male, `f` for female and `o`
for other, it wouldn't make sense for a mutated record to contain anything else except these three options.

Gecko offers the `with_categorical_values` function for this purpose.
It sources all possible options from a column in a CSV file and then applies random replacements respecting the limited
available options.

```python
import numpy as np
import pandas as pd

from gecko import mutator

rng = np.random.default_rng(22)
srs = pd.Series(["f", "m", "f", "f", "o", "m", "o", "o"])

categorical_mutator = mutator.with_categorical_values(
    "./gender.csv",  # CSV file containing "gender" column with "f", "m" and "o" as possible values
    value_column="gender",
    rng=rng,
)

print(categorical_mutator([srs]))
# => [["o", "f", "m", "o", "f", "o", "f", "m"]]
```

### Value permutations

Certain types of information are easily confused with others.
This is particularly true for names, where the differentiation between given and last names in a non-native language is
challenging to get right.
The `with_permute` function handles this exact use case.
It simply swaps the values between series that are passed into it.

```python
import pandas as pd

from gecko import mutator

srs_given_name = pd.Series(["Max", "Jane", "Jan"])
srs_last_name = pd.Series(["Mustermann", "Doe", "Jansen"])

permute_mutator = mutator.with_permute()
print(permute_mutator([srs_given_name, srs_last_name]))
# => [["Mustermann", "Doe", "Jansen"],
#       ["Max", "Jane", "Jan"]]
```

### Common replacements

Other various error sources, such as optical character recognition (OCR) errors, can be modeled using simple replacement
tables.
These tables have a source and a target column, defining mappings between character sequences.

The `with_replacement_table` function achieves just that.
Suppose you have the following CSV file with common OCR errors.

```csv
k,lc
5,s
2,z
1,|
```

You can use this file the same way you can with many other generation and mutation functions in Gecko.

```python
import numpy as np
import pandas as pd

from gecko import mutator

rng = np.random.default_rng(6379)
srs = pd.Series(["kick 0", "step 1", "go 2", "run 5"])

replacement_mutator = mutator.with_replacement_table(
    "./ocr.csv",
    rng=rng,
)

print(replacement_mutator([srs]))
# => ["lcick 0", "step |", "go z", "run s"]
```

## Multiple mutators

Using `mutate_data_frame`, you can apply multiple mutators on many columns at once.
It is possible to set probabilities for each mutator, as well as to define multiple mutators per column.

```python
import string

import numpy as np
import pandas as pd

from gecko import mutator

df = pd.DataFrame(
    {
        "fruit": ["apple", "banana", "orange"],
        "type": ["elstar", "cavendish", "mandarin"],
        "weight_in_grams": ["241.0", "195.6", "71.1"],
        "amount": ["3", "5", "6"],
        "grade": ["B", "C", "B"],
    }
)

rng = np.random.default_rng(25565)

df_mutated = mutator.mutate_data_frame(df, {
    ("fruit", "type"): (.5, mutator.with_permute()),  # (1)!
    "grade": [  # (2)!
        mutator.with_missing_value(strategy="all"),
        mutator.with_substitute(charset=string.ascii_uppercase, rng=rng),
    ],
    "amount": [  # (3)!
        (.8, mutator.with_insert(charset=string.digits, rng=rng)),
        (.2, mutator.with_delete(rng=rng))
    ]
}, rng=rng)

print(df_mutated)
# => [["fruit", "type", "weight_in_grams", "amount", "grade"],
#       ["apple", "elstar", "241.0", "83", ""],
#       ["cavendish", "banana", "195.6", "", "O"],
#       ["mandarin", "orange", "71.1", "", "F"]]
```

1. You can assign probabilities to a mutator for a column. In this case, the permutation mutator will be applied to
   50% of all records. The remaining 50% remain untouched.
2. You can assign multiple mutators to a column. In this case, the two mutators will be evenly applied to 50% of all
   records.
3. You can assign probabilities to multiple mutators for a column. In this case, the insertion and deletion mutator
   are applied to 80% and 20% of all records respectively.