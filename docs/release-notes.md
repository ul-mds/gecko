# Release notes

## 0.6.4 (Dec 6, 2024)

### Fixes

- Fix `with_regex_replacement_table` mutator not behaving correctly when pattern only matches partly

## 0.6.3 (Dec 4, 2024)

### Features

- Improve randomized selection of replacements in mutators using replacement tables

### Fixes

- Fix phonetic replacement rules not being matched correctly against original data when the desired pattern occurs in multiple places

## 0.6.2 (Nov 27, 2024)

### Features

- Add placeholder option to `with_generator` for inserting generated values

## 0.6.1 (Nov 15, 2024)

### Fixes

- Fix indexing behavior in `dfbitlookup` when using NumPy data types

## 0.6.0 (Nov 15, 2024)

### Breaking changes

- Change mutator type definition from `Callable[[list[pd.Series]], list[pd.Series]]` to `Callable[[list[pd.Series], Optional[float]], list[pd.Series]]` to delegate the selection of rows to mutate to the mutators themselves
- `Generator` and `Mutator` type definitions are now exported at the top level of the module
- Replace `D` option in favor of `d` for `unit` parameter in `with_datetime_offset`
- Remove `strategy` parameter from `with_missing_value`
- Remove `rng` parameter from `mutate_data_frame`
- Remove `with_edit` in favor of `with_group`

### Features

- `with_replacement_table`, `with_regex_replacement_table` and `with_phonetic_replacement_table` now favor rare replacements over common ones
- Add `rng` parameter to `with_function`, `with_lowercase`, `with_missing_value`, `with_noop`, `with_repeat`, `with_uppercase`
- `with_permute` now permutes series contents in a way that values are guaranteed to not remain in their original series
- Add `days`, `hours`, `minutes` and `seconds` to list of permitted `unit` values for `with_datetime_offset`
- Add `list[str]` as option to `charset` parameter of `with_cldr_keymap_file`, `with_insert` and `with_substitute`

### Fixes

- When providing a list of mutators to a column in `mutate_data_frame`, all mutators are now applied to all rows instead of with a `1 / mutator_count` probability
- Fix `with_regex_replacement_table` interpreting numbers in pattern and substitution columns as belonging to a named capture group 

### Documentation

- Use section-style navigation instead of tabs in Gecko docs

## 0.5.2 (Nov 5, 2024)

### Features

- Add `generator.with_group` for grouping multiple (weighted) generators together

### Internal

- Remove automated benchmarks

## 0.5.1 (Oct 30, 2024)

### Features

- Add the option to use data frames for all generators and mutators that accept paths to CSV files

## 0.5.0 (Oct 23, 2024)

### Breaking changes

- `to_data_frame` has a new call signature that ensures that it's consistent with `mutate_data_frame`

```python
df_generated = generator.to_data_frame(
    [
        (("fruit", "type"), generator.from_multicolumn_frequency_table(
            "fruit-types.csv",
            value_columns=["fruit", "type"],
            freq_column="count",
            rng=rng,
        )),
        ("weight", generator.from_uniform_distribution(
            low=20,
            high=100,
            rng=rng,
        )),
    ], 
    10_000
)
```

### Features

- Add `mutator.with_group` for grouping multiple mutators together
- Add support for Python 3.13

### Documentation

- Fix creation and modification timestamps in documentation

## 0.4.2 (Sep 20, 2024)

### Fixes

- Fix `NaN`s produced by generators and mutators that take in CSV files with empty cells

## 0.4.1 (Sep 12, 2024)

### Features

- Add `inline` and `reverse` flags to `with_replacement_table` mutator

## 0.4.0 (Sep 10, 2024)

### Breaking changes

- `mutate_data_frame` has a new call signature which ensures the order of mutation operations

```python
df_mutated = mutator.mutate_data_frame(
    df_original,
    [
        ("gender", (0.1, mutator.with_categorical_values(
            "./gender.csv",
            value_column="gender",
            rng=rng
        ))),
        (("given_name", "last_name"), (0.05, mutator.with_permute())),
        ("postcode", [
            mutator.with_delete(rng=rng),
            mutator.with_substitute(charset="0123456789", rng=rng)
        ])
    ],
    rng=rng
)
```

### Features

- Add `generator.from_datetime_range` for generating dates and times
- Add `mutator.with_lowercase` and `mutator.with_uppercase` for case conversions
- Add `mutator.with_datetime_offset` for applying arbitrary offsets to dates and times
- Add `mutator.with_generator` for appending, prepending or replacing data in a series with values from a generator
- Add `mutator.with_regex_replacement_table` for regex-based substitutions
- Add `mutator.with_repeat` for repeated values

### Fixes

- Fix `mutate_data_frame` raising an error if probability is provided as an integer, not a float

## 0.3.2 (Jul 19, 2024)

### Fixes

- Fix multiple mutators not being applied correctly when defined on the same column

## 0.3.1 (Mar 28, 2024)

### Fixes

- Fix `IndexError` when calling `with_permute` on empty series
- Fix Python version range in `pyproject.toml`

### Refactors

- Fix type hint on `**kwargs` in benchmarks

### Documentation

- Add navigation tabs to documentation
- Fix image link in README so that it can be displayed on PyPI

### Internal

- Cache dependencies in CI pipelines
- Reorganize dependencies into groups for tests, development and documentation

## 0.3.0 (Mar 18, 2024)

### Features

- Allow `corruptor.with_permute` to work with more than two series at once
- Infer `header` parameter for functions reading CSV files
- Remove list length constraints from `mutator` module

### Refactors

- Fix type hints on `*args` and `**kwargs`
- Rename `corruptor` module to `mutator`
- Rename `Corruptor` type alias to `Mutator`
- Rename `corruptor.corrupt_dataframe` to `mutator.mutate_data_frame`
- Rename `generator.to_dataframe` to `generator.to_dataframe`

### Documentation

- Add API reference to documentation
- Update docs to use new "mutator" terminology wherever possible
- Use Google format docstrings instead of reST

### Internal

- Merge documentation repository into main repository
- Move repositories from GitLab to GitHub
- Refine benchmark suite, add example based on [German population dataset](examples/german.md)

## 0.2.0 (Feb 16, 2024)

### Features

- Add `generator.with_permute` for swapping values between series
- Set wider version ranges for dependencies
- Fix `corruptor.corrupt_dataframe` to not modify original data frame
- Add tests to all corruptor functions to ensure no modifications to original data

### Refactors

- Change generators to take in and return a list of series instead of single series
- Change `generator.to_dataframe` signature to align with `corruptor.corrupt_dataframe`

### Internal

- Extend CI pipeline with a benchmarking step that runs on release and when manually triggered
- Add benchmark based on the "fruits" example in the docs

## 0.1.0 (Feb 8, 2024)

- Initial release