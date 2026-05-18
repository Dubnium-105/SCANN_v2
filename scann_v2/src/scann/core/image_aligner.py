"""Siril-only image alignment helpers for FITS pairs."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from scann.core.models import AlignResult


logger = logging.getLogger(__name__)

_SUPPORTED_METHODS = {"auto", "siril"}


def align(
    new_image: np.ndarray,
    old_image: np.ndarray,
    method: str = "siril",
    max_shift: int = 100,
) -> AlignResult:
    """Align ``old_image`` onto ``new_image`` using Siril only.

    ``method="auto"`` is kept as a compatibility alias for existing callers,
    but it still runs the same Siril-only path.
    """
    if method not in _SUPPORTED_METHODS:
        return AlignResult(
            aligned_old=None,
            success=False,
            error_message=f"Unsupported alignment method: {method}. Only Siril is available.",
        )

    if new_image.shape != old_image.shape:
        return AlignResult(
            aligned_old=None,
            success=False,
            error_message=f"Image dimensions do not match: new={new_image.shape}, old={old_image.shape}",
        )

    try:
        result, attempt_name, _original_score, _rotated_score = align_with_rot180_selection(
            new_image,
            old_image,
            method="siril",
            max_shift=max_shift,
            align_fn=_align_siril,
        )
    except Exception as exc:
        return AlignResult(aligned_old=None, success=False, error_message=str(exc))

    if not result.success or result.aligned_old is None:
        return result

    rotation = 180.0 if attempt_name == "rot180" else 0.0
    return _copy_result(result, rotation=rotation)


def align_with_rot180_selection(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    *,
    method: str = "siril",
    max_shift: int = 100,
    align_fn: Callable[..., AlignResult] | None = None,
) -> tuple[AlignResult, str, float, float]:
    """Try Siril alignment with the more plausible 180-degree orientation first."""
    if method not in _SUPPORTED_METHODS:
        return (
            AlignResult(
                aligned_old=None,
                success=False,
                error_message=f"Unsupported alignment method: {method}. Only Siril is available.",
            ),
            "original",
            float("nan"),
            float("nan"),
        )

    original_score, rotated_score = _orientation_scores(reference_image, moving_image)
    preferred = "rot180" if rotated_score > original_score + 1e-3 else "original"
    attempts = (
        ("rot180", np.rot90(moving_image, 2)),
        ("original", moving_image),
    ) if preferred == "rot180" else (
        ("original", moving_image),
        ("rot180", np.rot90(moving_image, 2)),
    )

    runner = _align_siril if align_fn is None or align_fn is align else align_fn
    failures: list[str] = []
    last_failure: AlignResult | None = None
    for attempt_name, attempt_image in attempts:
        try:
            result = runner(
                reference_image,
                attempt_image,
                method="siril",
                max_shift=max_shift,
            )
        except TypeError:
            result = runner(reference_image, attempt_image, max_shift=max_shift)
        except Exception as exc:
            result = AlignResult(aligned_old=None, success=False, error_message=str(exc))

        if result.success and result.aligned_old is not None:
            rotation = 180.0 if attempt_name == "rot180" else 0.0
            return _copy_result(result, rotation=rotation), attempt_name, original_score, rotated_score

        last_failure = result
        failures.append(f"{attempt_name}: {result.error_message or 'Siril alignment failed'}")

    return (
        AlignResult(
            aligned_old=None,
            dx=last_failure.dx if last_failure is not None else 0.0,
            dy=last_failure.dy if last_failure is not None else 0.0,
            success=False,
            error_message="; ".join(failures) or "Siril alignment failed for original and rot180",
        ),
        preferred,
        original_score,
        rotated_score,
    )


def choose_best_rot180_orientation(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    *,
    min_margin: float = 1e-3,
) -> tuple[np.ndarray, str, float, float]:
    """Choose between original and 180-degree rotated moving image for attempt ordering."""
    original_score, rotated_score = _orientation_scores(reference_image, moving_image)
    if rotated_score > original_score + min_margin:
        return np.rot90(moving_image, 2), "rot180", original_score, rotated_score
    return moving_image, "original", original_score, rotated_score


def _align_siril(
    new_image: np.ndarray,
    old_image: np.ndarray,
    method: str = "siril",
    max_shift: int = 100,
) -> AlignResult:
    """Align one orientation with Siril CLI."""
    _ = max_shift
    if method not in _SUPPORTED_METHODS:
        return AlignResult(
            aligned_old=None,
            success=False,
            error_message=f"Unsupported alignment method: {method}. Only Siril is available.",
        )

    executable = _find_siril_executable()
    if not executable:
        return AlignResult(
            aligned_old=None,
            success=False,
            error_message="Siril CLI not found (siril-cli/siril)",
        )

    from astropy.io import fits as astropy_fits

    with tempfile.TemporaryDirectory(prefix="scann_siril_align_") as temp_dir:
        work_dir = Path(temp_dir)
        ref_path = work_dir / "a_ref.fit"
        old_path = work_dir / "b_old.fit"
        script_path = work_dir / "align.ssf"

        new_sanitized = _sanitize_image_data(new_image)
        old_sanitized = _sanitize_image_data(old_image)
        astropy_fits.PrimaryHDU(data=new_sanitized).writeto(ref_path, overwrite=True)
        astropy_fits.PrimaryHDU(data=old_sanitized).writeto(old_path, overwrite=True)

        attempts = [
            ("1", "affine", "2.5", "0.25", "off"),
            ("1", "similarity", "1.6", "0.15", "off"),
            ("1", "shift", "1.2", "0.10", "off"),
        ]

        aligned_old_path: Path | None = None
        last_proc = None
        for setref_idx, transf, sigma, roundness, relax in attempts:
            _cleanup_siril_outputs(work_dir)
            script = "\n".join(
                [
                    "requires 1.2.0",
                    f'cd "{work_dir.as_posix()}"',
                    f"setfindstar reset -sigma={sigma} -roundness={roundness} -relax={relax}",
                    "link pair",
                    f"setref pair_ {setref_idx}",
                    f"register pair_ -transf={transf} -interp=lanczos4 -prefix=r_ -maxstars=2000",
                    "exit",
                    "",
                ]
            )
            script_path.write_text(script, encoding="utf-8")

            try:
                proc = subprocess.run(
                    [executable, "-d", str(work_dir), "-s", str(script_path)],
                    capture_output=True,
                    text=False,
                    timeout=120,
                    check=False,
                )
            except Exception as exc:
                logger.exception("Siril execution failed")
                return AlignResult(
                    aligned_old=None,
                    success=False,
                    error_message=f"Siril execution failed: {exc}",
                )

            last_proc = proc
            aligned_old_path = _find_siril_aligned_output(work_dir)
            if aligned_old_path is not None:
                logger.info(
                    "Siril alignment output found: %s (setref=%s, transf=%s)",
                    aligned_old_path,
                    setref_idx,
                    transf,
                )
                break

            logger.warning(
                "Siril attempt failed (setref=%s, transf=%s, rc=%s): %s | %s",
                setref_idx,
                transf,
                proc.returncode,
                _safe_decode(proc.stdout or b"")[-200:],
                _safe_decode(proc.stderr or b"")[-200:],
            )

        if aligned_old_path is None:
            produced = ", ".join(sorted(path.name for path in work_dir.iterdir()))
            out = _safe_decode((last_proc.stdout if last_proc else b"") or b"")
            err = _safe_decode((last_proc.stderr if last_proc else b"") or b"")
            return AlignResult(
                aligned_old=None,
                success=False,
                error_message=(
                    f"Siril did not produce aligned output; rc="
                    f"{(last_proc.returncode if last_proc else 'N/A')}; "
                    f"out={out[-500:]}; err={err[-500:]}; files={produced}"
                ),
            )

        aligned = _read_siril_output(astropy_fits, aligned_old_path)
        if aligned is None:
            return AlignResult(
                aligned_old=None,
                success=False,
                error_message=f"Failed to read Siril output: {aligned_old_path}",
            )

        return AlignResult(
            aligned_old=_match_intensity_scale(aligned, old_image),
            dx=0.0,
            dy=0.0,
            rotation=0.0,
            success=True,
        )


def _find_siril_executable() -> Optional[str]:
    """Find Siril CLI in PATH."""
    for name in ("siril-cli", "siril-cli.exe", "siril", "siril.exe"):
        executable = shutil.which(name)
        if executable:
            logger.info("Siril executable found: %s", executable)
            return executable
    logger.warning("Siril executable not found in PATH")
    return None


def _cleanup_siril_outputs(work_dir: Path) -> None:
    cleanup_patterns = [
        "pair*.fit",
        "pair*.fits",
        "pair*.fts",
        "pair*.seq",
        "pair_conversion.txt",
        "r_pair*.fit",
        "r_pair*.fits",
        "r_pair*.fts",
        "R_PAIR*.FIT",
        "R_PAIR*.FITS",
        "R_PAIR*.FTS",
    ]
    for pattern in cleanup_patterns:
        for path in work_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass

    cache_dir = work_dir / "cache"
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir, ignore_errors=True)


def _find_siril_aligned_output(work_dir: Path) -> Path | None:
    preferred = [
        work_dir / "r_pair_00002.fit",
        work_dir / "r_pair_00002.fits",
        work_dir / "r_pair_00002.fts",
        work_dir / "R_PAIR_00002.FIT",
        work_dir / "R_PAIR_00002.FITS",
        work_dir / "R_PAIR_00002.FTS",
    ]
    return next((path for path in preferred if path.is_file()), None)


def _read_siril_output(astropy_fits, aligned_old_path: Path) -> np.ndarray | None:
    last_exc: Exception | None = None
    for _ in range(10):
        try:
            if not aligned_old_path.is_file():
                time.sleep(0.2)
                continue
            with astropy_fits.open(str(aligned_old_path.resolve()), memmap=False) as hdul:
                data = hdul[0].data
            if data is None:
                last_exc = RuntimeError("Siril output is empty")
                time.sleep(0.2)
                continue
            return np.array(data, copy=True)
        except Exception as exc:
            last_exc = exc
            time.sleep(0.2)
    logger.warning("Read Siril output failed: %s", last_exc)
    return None


def _safe_decode(data: bytes) -> str:
    for encoding in ("utf-8", "gbk", "mbcs"):
        try:
            return data.decode(encoding)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def _sanitize_image_data(data: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(data, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _orientation_scores(reference_image: np.ndarray, moving_image: np.ndarray) -> tuple[float, float]:
    return (
        _orientation_similarity(reference_image, moving_image),
        _orientation_similarity(reference_image, np.rot90(moving_image, 2)),
    )


def _orientation_similarity(reference_image: np.ndarray, candidate_image: np.ndarray) -> float:
    """Cheap 180-degree orientation signal; no geometric alignment is performed."""
    if reference_image.shape[:2] != candidate_image.shape[:2]:
        return float("-inf")
    ref = _robust_normalize(reference_image)
    candidate = _robust_normalize(candidate_image)
    ref -= float(np.mean(ref))
    candidate -= float(np.mean(candidate))
    denom = float(np.linalg.norm(ref.ravel()) * np.linalg.norm(candidate.ravel()))
    if denom <= 1e-12:
        return float("-inf")
    return float(np.dot(ref.ravel(), candidate.ravel()) / denom)


def _robust_normalize(image: np.ndarray) -> np.ndarray:
    arr = _sanitize_image_data(image)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=2)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape[:2], dtype=np.float32)
    vals = arr[finite]
    p1, p99 = np.percentile(vals, [1, 99])
    if p99 <= p1:
        p1 = float(np.min(vals))
        p99 = float(np.max(vals))
    if p99 <= p1:
        return np.zeros(arr.shape[:2], dtype=np.float32)
    return ((np.clip(arr, p1, p99) - p1) / (p99 - p1)).astype(np.float32)


def _match_intensity_scale(aligned: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map normalized Siril output back to the reference brightness range when needed."""
    aligned_arr = _sanitize_image_data(aligned)
    ref_arr = _sanitize_image_data(reference)
    aligned_p1, aligned_p99 = np.percentile(aligned_arr, [1, 99])
    ref_p1, ref_p99 = np.percentile(ref_arr, [1, 99])
    aligned_range = float(aligned_p99 - aligned_p1)
    ref_range = float(ref_p99 - ref_p1)
    if aligned_range <= 1e-6:
        return aligned_arr

    ratio = ref_range / aligned_range if aligned_range > 0 else 1.0
    if 0.05 < ratio < 20.0:
        return aligned_arr

    mapped = (aligned_arr - aligned_p1) * (ref_range / aligned_range) + ref_p1
    mapped = np.clip(mapped, float(np.min(ref_arr)), float(np.max(ref_arr)))
    return mapped.astype(np.float32)


def _copy_result(result: AlignResult, *, rotation: float) -> AlignResult:
    return AlignResult(
        aligned_old=result.aligned_old,
        dx=float(result.dx),
        dy=float(result.dy),
        rotation=float(rotation),
        success=bool(result.success),
        error_message=str(result.error_message or ""),
    )


def batch_align(
    new_images: list[np.ndarray],
    old_images: list[np.ndarray],
    method: str = "siril",
    max_shift: int = 100,
) -> list[AlignResult]:
    """Batch-align image pairs using Siril only."""
    if len(new_images) != len(old_images):
        raise ValueError("new_images and old_images must have the same length")
    return [
        align(new_img, old_img, method=method, max_shift=max_shift)
        for new_img, old_img in zip(new_images, old_images)
    ]
