# Scripts Layout

This directory keeps non-package utilities together so the repository root stays
focused on product entry points.

## Folders

- `diagnostics/`: manual smoke tests and one-off verification scripts. These are
  intentionally kept out of `tests/` so `pytest` does not collect them by
  default.
- `legacy/`: legacy triplet-image training utilities that still belong to the
  old workflow, but are useful to keep around.

## Existing release scripts

- `build_exe_release.ps1`: build the Windows executable release bundle.
- `package_minimal_release.ps1`: assemble a minimal distributable package.
