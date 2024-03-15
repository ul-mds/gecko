# Release notes

## 0.3.0 (Mar 15, 2024)

### Features

- Infer `header` parameter from functions reading CSV files

### Refactors

- Rename `corruptor` module to `mutator`
- Rename `Corruptor` type alias to `Mutator`
- Rename `generator.to_dataframe` to `generator.to_dataframe`
- Rename `corruptor.corrupt_dataframe` to `mutator.mutate_data_frame`

### Documentation

- Add API reference
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