# GitHub Actions workflows

This folder contains the repository automation. Each YAML file owns one task:

- `gallery.yml` builds the Sphinx-Gallery example gallery from `examples/gallery/` and publishes it to GitHub Pages. It is path-filtered so README-only changes do not redeploy the gallery.
- `tests.yml` runs the notebook-backed visual regression test suite through `pytest test_suite/test_notebook_suite.py`. On failure, it uploads generated plots and diff plots as artifacts.
- `readme-toc.yml` checks that the README table of contents matches the README headings.

If a workflow breaks and blocks unrelated work, it can be temporarily disabled by renaming or removing its YAML file. Prefer fixing the failing workflow before merging release-facing changes.
