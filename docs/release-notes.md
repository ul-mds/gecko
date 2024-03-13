# Release notes

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