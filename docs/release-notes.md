# Release notes

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