# Test Layout

`tests/` is reserved for automated test suites collected by `pytest`.

- Put repeatable unit, integration, and regression tests here.
- Keep manual verification helpers in [`../scripts/diagnostics`](../scripts/README.md)
  so they do not get collected accidentally.
- Add subdirectories when a domain has enough tests to benefit from grouping,
  such as the existing `bridge/` suite.
