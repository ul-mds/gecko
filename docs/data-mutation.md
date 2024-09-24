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
Specifying the `inline` flag ensures that replacements are performed within words.

```python
import numpy as np
import pandas as pd

from gecko import mutator

rng = np.random.default_rng(6379)
srs = pd.Series(["kick 0", "step 1", "go 2", "run 5"])

replacement_mutator = mutator.with_replacement_table(
    "./ocr.csv",
    inline=True,
    rng=rng,
)

print(replacement_mutator([srs]))
# => ["lcick 0", "step |", "go z", "run s"]
```

To only replace whole words, leave out the `inline` flag or set it to `False`.
One use case is to replace names that sound or seem similar.

=== "CSV"

    ```csv
    source,target
    Jan,Jann
    Jan,Jean
    Jan,John
    Jan,Juan
    Jann,Jean
    Jann,Johann
    Jann,John
    Jann,Juan
    ```

=== "Table"

    | **Source** | **Target** | 
    | :-- | :-- |
    | Jan | Jann |
    | Jan | Jean |
    | Jan | John |
    | Jan | Juan |
    | Jann | Jean |
    | Jann | Johann |
    | Jann | John |
    | Jann | Juan |

Assuming the table shown above, one could perform randomized replacements using this mutator like so.

```python
import numpy as np
import pandas as pd

from gecko import mutator

rng = np.random.default_rng(6379)
srs = pd.Series(["Jan", "Jann", "Juan"])

replacement_mutator = mutator.with_replacement_table(
    "./given-names.csv",
    rng=rng,
)

print(replacement_mutator([srs]))
# => ["Jann", "John", "Juan"]
```

Note how "Juan" is not replaced since it is only present in the "target" column, not the "source" column.
By default, this mutator only considers replacement from the "source" to the "target" column.
If it should also consider reverse replacements, set the `reverse` flag.

```python
import numpy as np
import pandas as pd

from gecko import mutator

rng = np.random.default_rng(6379)
srs = pd.Series(["Jan", "Jann", "Juan"])

replacement_mutator = mutator.with_replacement_table(
    "./given-names.csv",
    reverse=True,
    rng=rng,
)

print(replacement_mutator([srs]))
# => ["Jann", "Juan", "Jan"]
```

### Regex replacements

Where the phonetic and generic replacement mutators do not fit the bill, replacements using regular expressions might
come in handy.
`with_regex_replacement_table` supports the application of mutations based on regular expressions.
This mutator works off of CSV files which contain the regular expression patterns to look for and the substitutions 
to perform as columns.

!!! warning

    Before using this mutator, make sure that `with_phonetic_replacement_table` and `with_replacement_table` are not
    suitable for your use case. 
    These functions are more optimised, whereas `with_regex_replacement_table` has to perform 
    replacements on a mostly row-by-row basis which impacts performance.

Let's assume that you want to perform mutations on a column containing dates where the digits of certain days
should be flipped.
A CSV file that is capable of these mutations could look as follows.

=== "CSV"

    ```csv
    pattern,1
    "\d{4}-\d{2}-(30)","03"
    "\d{4}-\d{2}-(20)","02"
    "\d{4}-\d{2}-(10)","01"
    ```

=== "Table"

    | **Pattern** | **1** | 
    | :-- | :-- |
    | `\d{4}-\d{2}-(30)` | `03` |
    | `\d{4}-\d{2}-(20)` | `02` |
    | `\d{4}-\d{2}-(10)` | `01` |

A mutator using the CSV file above would look for dates that have "10", "20" or "30" in their "day" field and flips
the digits to "01", "02" and "03" respectively.
This is done by placing a capture group around the "day" field in the regular expression.
Since it is the first capture group, once a row matches, Gecko will look up the substitution in the column labelled "1"
in the CSV file.
This also works when using named capture groups, in which case Gecko will use the name of the capture group to look up 
substitutions.

=== "CSV"

    ```csv
    pattern,day
    "\d{4}-\d{2}-(?P<day>30)","03"
    "\d{4}-\d{2}-(?P<day>20)","02"
    "\d{4}-\d{2}-(?P<day>10)","01"
    ```

=== "Table"

    | **Pattern** | **Day** | 
    | :-- | :-- |
    | `\d{4}-\d{2}-(?P<day>30)` | `03` |
    | `\d{4}-\d{2}-(?P<day>20)` | `02` |
    | `\d{4}-\d{2}-(?P<day>10)` | `01` |

Substitutions may also reference named capture groups.
Suppose you want to flip the least significant digit of the "day" and "month" field under certain conditions.
A CSV file capable of performing this type of substitution looks as follows.

=== "CSV"

    ```csv
    pattern,month,day
    "\d{4}-0(?P<month>[1-8])-[0-2](?P<day>[1-8])","(?P<day>)","(?P<month>)"
    ```

=== "Table"

    | **Pattern** | **Month** | **Day** | 
    | :-- | :-- | :-- |
    | `\d{4}-0(?P<month>[1-8])-[0-2](?P<day>[1-8])` | `(?P<day>)` | `(?P<month>)` |

`with_regex_replacement_table` works much like its "phonetic" and "common" siblings in that it requires a path to a CSV
file as shown above and the name of the column containing the regex patterns to look for.
The columns containing the substitution values are inferred at runtime.
In the following snippet, the second example using named capture groups to flip the digits in the day field is shown.

```python
import numpy as np
import pandas as pd

from gecko import mutator

rng = np.random.default_rng(0x2321)
srs = pd.Series(["2020-01-30", "2020-01-20", "2020-01-10"])

regex_mutator = mutator.with_regex_replacement_table(
    "./dob-day-digit-flip.csv",
    pattern_column="pattern",
    rng=rng
)

print(regex_mutator([srs]))
# => ["2020-01-03", "2020-01-02", "2020-01-01"]
```

It is also possible to define a column that contains [regex flags](https://docs.python.org/3/library/re.html#flags).
At the time, Gecko supports the `ASCII` and `IGNORECASE` flags which can be applied by adding `a` and `i` respectively 
to the flag column.

=== "CSV"

    ```csv
    pattern,suffix,flags
    "fooba(?P<suffix>r)","z","i"
    ```

=== "Table"

    | **Pattern** | **Suffix** | **Flags** | 
    | :-- | :-- | :-- |
    | `fooba(?P<suffix>r)` | `z` | `i` |

In the following snippet, case-insensitive matching will be performed.
This causes all rows of the input series to be modified.

```python
import numpy as np
import pandas as pd

from gecko import mutator

rng = np.random.default_rng(0xCAFED00D)
srs = pd.Series(["foobar", "Foobar", "fOoBaR"])

regex_mutator = mutator.with_regex_replacement_table(
    "./foobar.csv",
    pattern_column="pattern",
    flags_column="flags",
    rng=rng
)

print(regex_mutator([srs]))
# => ["foobaz", "Foobaz", "fOoBaz"]
```

### Case conversions

During data entry or normalization, it may occur that text is converted to all lowercase or uppercase, by accident or on
purpose.
`with_lowercase` and `with_uppercase` handle these use cases.

```python
import pandas as pd

from gecko import mutator

srs = pd.Series(["Foobar", "foobaz", "FOOBAT"])

lowercase_mutator = mutator.with_lowercase()
uppercase_mutator = mutator.with_uppercase()

print(lowercase_mutator([srs]))
# => ["foobar", "foobaz", "foobat"]
print(uppercase_mutator([srs]))
# => ["FOOBAR", "FOOBAZ", "FOOBAT"]
```

### Date and time offsets

Date and time information is prone to errors where single fields are offset by a couple units.
This error source is implemented in the `with_datetime_offset` function.
It requires a range in which time units can be offset by and the format of the data to mutate as expressed
by [Python's format codes for datetime objects](https://docs.python.org/3/library/datetime.html#format-codes).
It is possible to apply offsets in units of days (`D`), hours (`h`), minutes (`m`) and seconds (`s`).

```python
import numpy as np
import pandas as pd

from gecko import mutator

srs = pd.Series(pd.date_range("2020-01-01", "2020-01-31", freq="D"))
rng = np.random.default_rng(0xffd8)

datetime_mutator = mutator.with_datetime_offset(
   max_delta=5, unit="D", dt_format="%Y-%m-%d", rng=rng
)

print(datetime_mutator([srs]))
# => ["2019-12-30", "2020-01-03", ..., "2020-01-25", "2020-02-01"]
```

When applying offsets, it might happen that the offset applied to a single field affects another field, e.g. subtracting
a day from January 1st, 2020 will wrap around to December 31st, 2019.
If this is not desired, Gecko offers an extra flag that prevents these types of wraparounds at the cost of leaving 
affected rows untouched.
Note how the first and last entry in the output of the snippet below remains unchanged when compared to the previous 
snippet.

```python
import numpy as np
import pandas as pd

from gecko import mutator

srs = pd.Series(pd.date_range("2020-01-01", "2020-01-31", freq="D"))
rng = np.random.default_rng(0xffd8)

datetime_mutator = mutator.with_datetime_offset(
   max_delta=5, unit="D", dt_format="%Y-%m-%d", prevent_wraparound=True, rng=rng
)

print(datetime_mutator([srs]))
# => ["2020-01-01", "2020-01-03", ..., "2020-01-25", "2020-01-31"]
```

### Repeated values

Erroneous copy-paste operations may yield an unwanted duplication of values.
This is implemented in Gecko's `with_repeat` mutator.
By default, it appends values with a space, but a custom joining character can be defined as well.

```python
import numpy as np
import pandas as pd

from gecko import mutator

srs = pd.Series(["foo", "bar", "baz"])

repeat_mutator = mutator.with_repeat()
repeat_mutator_no_space = mutator.with_repeat(join_with="")

print(repeat_mutator([srs]))
# => ["foo foo", "bar bar", "baz baz"]

print(repeat_mutator_no_space([srs]))
# => ["foofoo", "barbar", "bazbaz"]
```

### Using generators

`with_generator` can leverage Gecko's mutators to prepend, append or replace data.
For instance, this can be used for emulating compound names for persons who have more than one given or last name.
By default, this function adds a space when prepending or appending generated data, but this can be customized.

```python
import numpy as np
import pandas as pd

from gecko import mutator


def generate_foobar_suffix(rand: np.random.Generator):
    def _generate(count: int) -> list[pd.Series]:
        return [pd.Series(rand.choice(("bar", "baz", "bat"), size=count))]

    return _generate


srs = pd.Series(["foo"] * 100)
rng = np.random.default_rng(0x25504446)

gen_prepend_mutator = mutator.with_generator(generate_foobar_suffix(rng), "prepend")
print(gen_prepend_mutator([srs]))
# => ["bat foo", "bar foo", ..., "baz foo", "baz foo"]

gen_replace_mutator = mutator.with_generator(generate_foobar_suffix(rng), "replace")
print(gen_replace_mutator([srs]))
# => ["bar", "bar", ..., "baz", "bat"]

gen_append_mutator = mutator.with_generator(generate_foobar_suffix(rng), "append", join_with="")
print(gen_append_mutator([srs]))
# => ["foobat", "foobat", ..., "foobat", "foobaz"]
```

### Grouped mutators

When applying mutators that are mutually exclusive, `with_group` can be used.
It can take a list of mutators or a list of weighted mutators as arguments.
When providing a list of mutators, all mutators are applied with equal probability.
When using weighted mutators, each mutator is applied with its assigned probability.

```python
import numpy as np
import pandas as pd

from gecko import mutator

rng = np.random.default_rng(123)
srs = pd.Series(["a"] * 100)

equal_prob_mutator = mutator.with_group([
   mutator.with_insert(rng=rng),
   mutator.with_delete(rng=rng),
], rng=rng)

(srs_mut_1,) = equal_prob_mutator([srs])
print(srs_mut_1.str.len().value_counts())
# => { 0: 45, 2: 55 }
# no more single character values remain

weighted_prob_mutator = mutator.with_group([
   (.25, mutator.with_insert(rng=rng)),
   (.25, mutator.with_delete(rng=rng)),
], rng=rng)

(srs_mut_2,) = weighted_prob_mutator([srs])
print(srs_mut_2.str.len().value_counts())
# => { 0: 24, 1: 50, 2: 26 }
# half of the original single character values remain
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

df_mutated = mutator.mutate_data_frame(df, [
    (("fruit", "type"), (.5, mutator.with_permute())),  # (1)!
    ("grade", [  # (2)!
        mutator.with_missing_value(strategy="all"),
        mutator.with_substitute(charset=string.ascii_uppercase, rng=rng),
    ]),
    ("amount", [  # (3)!
        (.8, mutator.with_insert(charset=string.digits, rng=rng)),
        (.2, mutator.with_delete(rng=rng))
    ])
], rng=rng)

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