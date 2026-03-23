# Repository Layout

This project now follows a simple placement rule:

- product code lives under `scann_v2/src/`
- automated tests live under `scann_v2/tests/`
- supporting documents live under `scann_v2/docs/`
- operational and diagnostic scripts live under `scann_v2/scripts/`

## Script categories

- `scripts/diagnostics/`: manual smoke tests and developer diagnostics.
- `scripts/legacy/`: legacy triplet-image training helpers.
- `scripts/*.ps1`: packaging and release automation used by the current v2 app.

## Why this split exists

- `pytest` only scans `tests/`, so manual scripts stay out of automated runs.
- legacy tools remain available without mixing them into the main app surface.
- release scripts, diagnostics, and docs all have a predictable home.

## Quick examples

```powershell
cd scann_v2
python .\scripts\diagnostics\logging_smoke.py
python .\scripts\legacy\calc_triplet_mean_std.py --neg ..\dataset\negative --pos ..\dataset\positive
powershell -ExecutionPolicy Bypass -File .\scripts\build_exe_release.ps1 -Clean
```
