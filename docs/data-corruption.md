# Corrupting data

Gecko comes with a lot of built-in functions that corrupt data by applying errors that one might find in the real world.
Any function that takes in and returns a list of [Pandas series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html) is considered a corruptor function in Gecko.

```py
import pandas as pd

def corruptor_func(srs_lst: list[pd.Series]) -> list[pd.Series]:
    assert len(srs_lst) == 1  # check amount of columns
    srs_out = srs_lst[0].copy()
    # ... perform mutations on the copy ...
    return [srs_out]
```

Gecko provides functions for introducing typographic, phonetic and other errors.
These functions are exposed in Gecko's `corruptor` module.

## Available corruptors

### Keyboard typos

One of the most common sources for typos are adjacent keys on a keyboard.
Gecko supports loading of keyboard layouts and applying typos based on them.
Currently, keyboard layouts must be provided as an XML file from the [Unicode CLDR repository](https://github.com/unicode-org/cldr).
Gecko parses these files and determines all neighboring keys of each key, as well as their variants with and without ++shift++ pressed.

!!! warning

    As of Unicode CLDR keyboard specification is under a major redesign as of release 44.
    Support will be added as soon as the specification is finalized.
    For now, please retrieve CLDR keyboard files from a release tagged 43 or earlier.
    The examples in this documentation use files from the CLDR release 43.

[Download the German keyboard layout from the CLDR repository](https://github.com/unicode-org/cldr/blob/release-43/keyboards/windows/de-t-k0-windows.xml).
The corresponding corruptor function is called `with_cldr_keymap_file`.
Point the corruptor to the file you just downloaded.
In the following example, one character in each word is substituted by another neighboring character on the German keyboard.

```py
import pandas as pd
import numpy as np

from gecko import corruptor

rng = np.random.default_rng(3141)
kb_corruptor = corruptor.with_cldr_keymap_file(
    "./de-t-k0-windows.xml",
    rng=rng
)
srs = pd.Series(["apple", "banana", "clementine"])
print(kb_corruptor([srs]))
# => [["spple", "banany", "cldmentine"]]
```

By default, this corruptor considers all possible neighboring keys for each key.
If you want to constrain typos to a certain set of characters, you can pass an optional string of characters to this corruptor.
One such example is to limit the corruptor to digits when manipulating a series of numbers that are broken up by non-digit characters.
The following snippet avoids the substitution of hyphens by specifying that only digits may be manipulated.

```py
import pandas as pd
import numpy as np
import string

from gecko import corruptor

rng = np.random.default_rng(2718)
kb_corruptor = corruptor.with_cldr_keymap_file(
    "./de-t-k0-windows.xml",
    charset=string.digits,
    rng=rng
)
srs = pd.Series(["123-456-789", "727-727-727", "294-753-618"])
print(kb_corruptor([srs]))
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
The signature is similar to that of `with_replacement_table`.

```py
import numpy as np
import pandas as pd

from gecko import corruptor

rng = np.random.default_rng(8844167)

phonetic_corruptor = corruptor.with_phonetic_replacement_table(
    "./phonetic-rules-de.csv",
    header=True,
    source_column="source",
    target_column="target",
    flags_column="flags",
    rng=rng,
)

srs = pd.Series(["straße", "stadt", "schießen"])
print(phonetic_corruptor([srs]))
# => [["strasse", "statt", "schiessen"]]
```

### Missing values

A textual representation of a "missing value" is sometimes used to clearly indicate that a blank or an empty value is to be interpreted as a missing piece of information.
In datasets sourced from large databases, this "missing value" might consist of characters that do not adhere to a table or column schema.
A simple example would be `###_MISSING_###` in place of a person's date of birth, since it does not conform to any common date format and consists entirely of letters and special characters. 

Gecko provides the function `with_missing_value` which replaces certain values within a series with a custom "missing value".
The corruptor replaces either empty, blank or all strings within a series depending on the defined strategy.
This is best explained by a few examples.

Gecko considers strings to be "empty" when their length is zero.
Strings with whitespaces will be left as-is.

```py
import pandas as pd

from gecko import corruptor

missing_corruptor = corruptor.with_missing_value(
    "###_MISSING_###", 
    strategy="empty"
)
srs = pd.Series(["apple", "   ", ""])
print(missing_corruptor([srs]))
# => [["apple", "   ", "###_MISSING_###"]]
```

Gecko considers strings to be "blank" when their length is zero after trimming all leading and trailing whitespaces.
This is the default behavior of this corruptor.

```py
import pandas as pd

from gecko import corruptor

missing_corruptor = corruptor.with_missing_value(
    "###_MISSING_###", 
    strategy="blank"
)
srs = pd.Series(["apple", "   ", ""])
print(missing_corruptor([srs]))
# => [["apple", "###_MISSING_###", "###_MISSING_###"]]
```

The "nuclear" option is to replace all strings within a series with the "missing value".

```py
import pandas as pd

from gecko import corruptor

missing_corruptor = corruptor.with_missing_value(
    "###_MISSING_###", 
    strategy="all"
)
srs = pd.Series(["apple", "   ", ""])
print(missing_corruptor([srs]))
# => [["###_MISSING_###", "###_MISSING_###", "###_MISSING_###"]]
```

### Edit errors

Edit errors are caused by a set of operations on single characters within a word.
There are commonly four operations that can induce these types of errors: insertion and deletion of a single character, substitution of a character with a different one, and transposition of two adjacent characters.

Gecko provides corruptors for each of these operations. 
For insertions and substitutions, it is possible to define a set of characters to choose from.

```python
import string

import numpy as np
import pandas as pd

from gecko import corruptor

rng = np.random.default_rng(8080)
srs = pd.Series(["apple", "banana", "clementine"])
    
insert_corruptor = corruptor.with_insert(charset=string.ascii_letters, rng=rng)
print(insert_corruptor([srs]))
# => [["aVpple", "banaFna", "clemenMtine"]]

delete_corruptor = corruptor.with_delete(rng=rng)
print(delete_corruptor([srs]))
# => [["aple", "bnana", "clementin"]]

substitute_corruptor = corruptor.with_substitute(charset=string.digits, rng=rng)
print(substitute_corruptor([srs]))
# => [["appl9", "ba4ana", "clementi9e"]]

transpose_corruptor = corruptor.with_transpose(rng)
print(transpose_corruptor([srs]))
# => [["paple", "baanna", "clemenitne"]]
```

Gecko also provides a more general edit corruptor which wraps around the insertion, deletion, substitution and transposition corruptor.
It is then possible to assign probabilities for each operation.
By default, all operations are equally likely to be performed.

```python
import numpy as np
import pandas as pd

from gecko import corruptor

rng = np.random.default_rng(8443)
srs = pd.Series(["apple", "banana", "clementine", "durian", "eggplant", "fig", "grape", "honeydew"])

edit_corruptor_1 = corruptor.with_edit(rng=rng)
print(edit_corruptor_1([srs]))
# => [["aple", "banan", "clementinb", "duiran", "eAgplant", "Nig", "grapce", "hoKeydew"]]

edit_corruptor_2 = corruptor.with_edit(
    p_insert=0.1,
    p_delete=0.2,
    p_substitute=0.3,
    p_transpose=0.4,
    rng=rng,
)
print(edit_corruptor_2([srs]))
# => [["aplpe", "anana", "lementine", "duriRan", "geggplant", "fg", "rgape", "honedyew"]]
```

### Categorical errors

Sometimes an attribute can only take on a set number of values.
For example, if you have a "gender" column in your dataset and it can only take on `m` for male, `f` for female and `o` for other, it wouldn't make sense for a corrupted record to contain anything else except these three options.

Gecko offers the `with_categorical_values` function for this purpose.
It sources all possible options from a column in CSV file and then applies random replacements respecting the limited available options.

```python
import numpy as np
import pandas as pd

from gecko import corruptor

rng = np.random.default_rng(22)
srs = pd.Series(["f", "m", "f", "f", "o", "m", "o", "o"])

categorical_corruptor = corruptor.with_categorical_values(
    "./gender.csv",  # CSV file containing "gender" column with "f", "m" and "o" as possible values
    header=True,
    value_column="gender",
    rng=rng,
)

print(categorical_corruptor([srs]))
# => [["o", "f", "m", "o", "f", "o", "f", "m"]]
```

### Value permutations

Certain types of information are easily confused with others.
This is particularly true for names, where the differentiation between given and last names in a non-native language is challenging to get right.
The `with_permute` function handles this exact use case.
It simply swaps the values of two series that are passed into it.

```python
import pandas as pd

from gecko import corruptor

srs_given_name = pd.Series(["Max", "Jane", "Jan"])
srs_last_name = pd.Series(["Mustermann", "Doe", "Jansen"])

permute_corruptor = corruptor.with_permute()
print(permute_corruptor([srs_given_name, srs_last_name]))
# => [["Mustermann", "Doe", "Jansen"],
#       ["Max", "Jane", "Jan"]]
```

### Common replacements

Other various error sources, such as optical character recognition (OCR) errors, can be modeled using simple replacement tables.
These tables have a source and a target column, defining mappings between character sequences.

The `with_replacement_table` function achieves just that.
Suppose you have the following CSV file with common OCR errors.

```csv
k,lc
5,s
2,z
1,|
```

You can use this file the same way you can with many other generation and corruption functions in Gecko.

```python
import numpy as np
import pandas as pd

from gecko import corruptor

rng = np.random.default_rng(6379)
srs = pd.Series(["kick 0", "step 1", "go 2", "run 5"])

replacement_corruptor = corruptor.with_replacement_table(
    "./ocr.csv",
    rng=rng,
)

print(replacement_corruptor([srs]))
# => ["lcick 0", "step |", "go z", "run s"]
```

## Multiple corruptors

Using `corrupt_dataframe`, you can apply multiple corruptors on many columns at once.
It is possible to set probabilities for each corruptor, as well as to define multiple corruptors per column.

```python
import string

import numpy as np
import pandas as pd

from gecko import corruptor
from gecko.corruptor import with_permute

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

df_corrupt = corruptor.corrupt_dataframe(df, {
    ("fruit", "type"): (.5, with_permute()),  # (1)!
    "grade": [  # (2)!
        corruptor.with_missing_value(strategy="all"),
        corruptor.with_substitute(charset=string.ascii_uppercase, rng=rng),
    ],
    "amount": [  # (3)!
        (.8, corruptor.with_insert(charset=string.digits, rng=rng, )),
        (.2, corruptor.with_delete(rng=rng, ))
    ]
}, rng=rng)

print(df_corrupt)
# => [["fruit", "type", "weight_in_grams", "amount", "grade"],
#       ["apple", "elstar", "241.0", "83", ""],
#       ["cavendish", "banana", "195.6", "", "O"],
#       ["mandarin", "orange", "71.1", "", "F"]]
```

1. You can assign probabilities to a corruptor for a column. In this case, the permutation corruptor will be applied to 50% of all records. The remaining 50% remain untouched.
2. You can assign multiple corruptors to a column. In this case, the two corruptors will be evenly applied to 50% of all records.
3. You can assign probabilities to multiple corruptors for a column. In this case, the insertion and deletion corruptor are applied to 80% and 20% of all records respectively.