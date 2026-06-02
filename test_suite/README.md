# Test suite

The test suite is notebook-backed so image differences can be inspected inline when a plot changes.

Files in this folder:

- `test_suite.ipynb`: interactive source of the visual regression suite.
- `test_notebook_suite.py`: pytest wrapper used by GitHub Actions.
- `expected_plots/`: approved reference images.
- `test_plots/` and `diff_plots/`: generated local outputs; these are ignored by Git.

Run the automated suite from the repository root:

```shell
pytest test_suite/test_notebook_suite.py
```

If a visual change is intentional, review the generated output first and then replace the relevant file in `expected_plots/`.
