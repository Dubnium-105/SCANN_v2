"""Image alignment helpers for FITS pairs."""

from __future__ import annotations

from typing import Callable, List, Optional
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from scann.core.models import AlignResult


logger = logging.getLogger(__name__)


def align(
    new_image: np.ndarray,
    old_image: np.ndarray,
    method: str = "auto",
    max_shift: int = 100,
) -> AlignResult:
    """Align ``old_image`` onto ``new_image``."""
    if new_image.shape != old_image.shape:
        return AlignResult(
            aligned_old=None,
            success=False,
            error_message=f"图像尺寸不匹配: new={new_image.shape}, old={old_image.shape}",
        )

    try:
        if method == "auto":
            phase_result = _align_phase_correlation(new_image, old_image, max_shift)
            if phase_result.success:
                return phase_result

            ecc_result = _align_ecc(new_image, old_image, max_shift)
            if ecc_result.success:
                return ecc_result

            feature_result = _align_feature_matching(new_image, old_image, max_shift)
            if feature_result.success:
                return feature_result

            siril_result = _align_siril(new_image, old_image, max_shift)
            if siril_result.success:
                return siril_result

            return AlignResult(
                aligned_old=None,
                success=False,
                error_message=(
                    f"phase失败: {phase_result.error_message}; "
                    f"ECC失败: {ecc_result.error_message}; "
                    f"feature失败: {feature_result.error_message}"
                ),
            )

        if method == "phase_correlation":
            return _align_phase_correlation(new_image, old_image, max_shift)

        if method == "feature_matching":
            return _align_feature_matching(new_image, old_image, max_shift)

        if method == "ecc":
            return _align_ecc(new_image, old_image, max_shift)

        if method == "siril":
            return _align_siril(new_image, old_image, max_shift)

        return AlignResult(
            aligned_old=None,
            success=False,
            error_message=f"不支持的对齐方法: {method}",
        )
    except Exception as exc:
        return AlignResult(
            aligned_old=None,
            success=False,
            error_message=str(exc),
        )


def _to_gray_f32(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale float32 and sanitize NaN / Inf."""
    import cv2

    if image.ndim == 3:
        gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.float32)
    return np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_for_alignment(gray_f32: np.ndarray) -> np.ndarray:
    """Robustly normalize to ``[0, 1]`` for alignment."""
    finite = np.isfinite(gray_f32)
    if not np.any(finite):
        return np.zeros_like(gray_f32, dtype=np.float32)

    vals = gray_f32[finite]
    p1, p99 = np.percentile(vals, [1, 99])
    if p99 <= p1:
        p1 = float(np.min(vals))
        p99 = float(np.max(vals))
    if p99 <= p1:
        return np.zeros_like(gray_f32, dtype=np.float32)

    clipped = np.clip(gray_f32, p1, p99)
    norm = (clipped - p1) / (p99 - p1)
    return norm.astype(np.float32)


def _enhance_stars(norm01: np.ndarray) -> np.ndarray:
    """High-pass star enhancement."""
    import cv2

    low = cv2.GaussianBlur(norm01, (0, 0), sigmaX=2.0)
    high = norm01 - low
    high = np.clip(high, 0.0, None)
    max_val = float(np.max(high))
    if max_val > 0:
        high /= max_val
    return high.astype(np.float32)


def _warp_translate(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    import cv2

    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _zncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalized cross-correlation in ``[-1, 1]``."""
    aa = a.astype(np.float32).ravel()
    bb = b.astype(np.float32).ravel()
    aa -= float(np.mean(aa))
    bb -= float(np.mean(bb))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return -1.0
    return float(np.dot(aa, bb) / denom)


def _calc_overlap_bounds(width: int, height: int, dx: float, dy: float) -> tuple[int, int, int, int] | None:
    x0 = max(0, int(np.ceil(dx)))
    x1 = min(width, int(np.floor(width + dx)))
    y0 = max(0, int(np.ceil(dy)))
    y1 = min(height, int(np.floor(height + dy)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, x1, y0, y1


def _alignment_quality(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    aligned_image: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[float, float]:
    """Measure before / after similarity on the overlap region."""
    ref_n = _enhance_stars(_normalize_for_alignment(_to_gray_f32(reference_image)))
    mov_n = _enhance_stars(_normalize_for_alignment(_to_gray_f32(moving_image)))
    aligned_n = _enhance_stars(_normalize_for_alignment(_to_gray_f32(aligned_image)))

    bounds = _calc_overlap_bounds(ref_n.shape[1], ref_n.shape[0], dx, dy)
    if bounds is None:
        return _zncc(ref_n, mov_n), -1.0

    x0, x1, y0, y1 = bounds
    if (x1 - x0) < 32 or (y1 - y0) < 32:
        return _zncc(ref_n, mov_n), -1.0

    ref_crop = ref_n[y0:y1, x0:x1]
    mov_crop = mov_n[y0:y1, x0:x1]
    aligned_crop = aligned_n[y0:y1, x0:x1]
    return _zncc(ref_crop, mov_crop), _zncc(ref_crop, aligned_crop)


def _alignment_similarity(
    reference_image: np.ndarray,
    candidate_image: np.ndarray,
) -> float:
    """Measure full-frame similarity for orientation selection."""
    ref_n = _enhance_stars(_normalize_for_alignment(_to_gray_f32(reference_image)))
    candidate_n = _enhance_stars(_normalize_for_alignment(_to_gray_f32(candidate_image)))
    return _zncc(ref_n, candidate_n)


def choose_best_rot180_orientation(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    *,
    min_margin: float = 1e-3,
) -> tuple[np.ndarray, str, float, float]:
    """Choose between original and 180-degree rotated moving image."""
    original_score = _alignment_similarity(reference_image, moving_image)
    rotated_image = np.rot90(moving_image, 2)
    rotated_score = _alignment_similarity(reference_image, rotated_image)
    if rotated_score > original_score + min_margin:
        return rotated_image, "rot180", original_score, rotated_score
    return moving_image, "original", original_score, rotated_score


def align_with_rot180_selection(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    *,
    method: str = "auto",
    max_shift: int = 100,
    align_fn: Callable[..., AlignResult] = align,
) -> tuple[AlignResult, str, float, float]:
    """Align after selecting the more plausible 180-degree orientation."""
    preferred_image, preferred_name, original_score, rotated_score = choose_best_rot180_orientation(
        reference_image,
        moving_image,
    )
    fallback_image = np.rot90(moving_image, 2) if preferred_name == "original" else moving_image
    fallback_name = "rot180" if preferred_name == "original" else "original"

    last_failure: AlignResult | None = None
    for attempt_name, attempt_image in (
        (preferred_name, preferred_image),
        (fallback_name, fallback_image),
    ):
        try:
            result = align_fn(
                reference_image,
                attempt_image,
                method=method,
                max_shift=max_shift,
            )
        except Exception as exc:
            last_failure = AlignResult(
                aligned_old=None,
                success=False,
                error_message=str(exc),
            )
            continue

        if getattr(result, "success", False) and getattr(result, "aligned_old", None) is not None:
            return result, attempt_name, original_score, rotated_score
        last_failure = result

    return (
        last_failure
        or AlignResult(
            aligned_old=None,
            success=False,
            error_message="rot180 orientation alignment failed",
        ),
        preferred_name,
        original_score,
        rotated_score,
    )


def _is_quality_improved(
    before: float,
    after: float,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    min_delta: float = 5e-4,
    min_after: float = 5e-2,
    small_shift_px: float = 1.0,
    max_drop_small_shift: float = 2e-2,
) -> bool:
    if not np.isfinite(before) or not np.isfinite(after):
        return False
    if after < min_after:
        return False

    if after > before + min_delta:
        return True

    shift = float(np.hypot(dx, dy))
    if shift <= small_shift_px and after >= before - max_drop_small_shift:
        return True

    return False


def _match_intensity_scale(aligned: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map normalized Siril output back to the reference brightness range when needed."""
    aligned_arr = np.nan_to_num(aligned.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ref_arr = np.nan_to_num(reference.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

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


def _estimate_translation(reference_image: np.ndarray, moving_image: np.ndarray) -> tuple[float, float] | None:
    """Estimate the translation that should be applied to ``moving_image``."""
    import cv2

    ref = _enhance_stars(_normalize_for_alignment(_to_gray_f32(reference_image)))
    mov = _enhance_stars(_normalize_for_alignment(_to_gray_f32(moving_image)))
    height, width = ref.shape[:2]
    if height < 16 or width < 16:
        return None

    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(ref, mov, window)
    if not np.isfinite(dx) or not np.isfinite(dy):
        return None
    if response < 1e-4:
        return None

    # phaseCorrelate reports the opposite sign of the warp we later apply.
    return float(-dx), float(-dy)


def _align_phase_correlation(
    new_image: np.ndarray,
    old_image: np.ndarray,
    max_shift: int,
) -> AlignResult:
    """Translation-only alignment using phase correlation."""
    import cv2

    new_norm = _enhance_stars(_normalize_for_alignment(_to_gray_f32(new_image)))
    old_norm = _enhance_stars(_normalize_for_alignment(_to_gray_f32(old_image)))
    height, width = new_norm.shape[:2]

    candidates: list[tuple[float, float, float]] = []
    for frac in (1.0, 0.9, 0.8):
        crop_h = max(32, int(round(height * frac)))
        crop_w = max(32, int(round(width * frac)))
        y0 = (height - crop_h) // 2
        x0 = (width - crop_w) // 2
        new_crop = new_norm[y0:y0 + crop_h, x0:x0 + crop_w]
        old_crop = old_norm[y0:y0 + crop_h, x0:x0 + crop_w]
        window = cv2.createHanningWindow((crop_w, crop_h), cv2.CV_32F)
        (raw_dx, raw_dy), response = cv2.phaseCorrelate(new_crop, old_crop, window)
        if not np.isfinite(raw_dx) or not np.isfinite(raw_dy) or response < 1e-4:
            continue
        dx = float(-raw_dx)
        dy = float(-raw_dy)
        if abs(dx) <= max_shift and abs(dy) <= max_shift:
            candidates.append((dx, dy, float(response)))

    if not candidates:
        return AlignResult(
            aligned_old=None,
            success=False,
            error_message="相位相关未得到有效候选位移",
        )

    best_candidate: tuple[float, float, float, float, float] | None = None
    for dx, dy, response in candidates:
        aligned = _warp_translate(old_image, dx, dy)
        before, after = _alignment_quality(new_image, old_image, aligned, dx, dy)
        candidate = (after - before, after, response, dx, dy)
        if best_candidate is None or candidate > best_candidate:
            best_candidate = candidate

    assert best_candidate is not None
    improvement, after, response, dx, dy = best_candidate
    aligned = _warp_translate(old_image, dx, dy)
    before, _ = _alignment_quality(new_image, old_image, aligned, dx, dy)

    if not _is_quality_improved(before, after, dx=dx, dy=dy):
        return AlignResult(
            aligned_old=None,
            dx=dx,
            dy=dy,
            success=False,
            error_message=(
                f"相位相关质量不足: before={before:.4f}, after={after:.4f}, "
                f"response={response:.4f}"
            ),
        )

    return AlignResult(aligned_old=aligned, dx=dx, dy=dy, success=True)


def _align_ecc(
    new_image: np.ndarray,
    old_image: np.ndarray,
    max_shift: int,
) -> AlignResult:
    """ECC fallback with consistent transform direction."""
    import cv2

    new_norm = _normalize_for_alignment(_to_gray_f32(new_image))
    old_norm = _normalize_for_alignment(_to_gray_f32(old_image))
    initial_shift = _estimate_translation(new_image, old_image)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        200,
        1e-6,
    )

    best_result: AlignResult | None = None
    best_improvement = float("-inf")
    last_error = "ECC 收敛失败"

    for motion in (cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN):
        seeds = [None]
        if initial_shift is not None:
            seeds.append(initial_shift)

        for seed in seeds:
            try:
                warp = np.eye(2, 3, dtype=np.float32)
                if seed is not None:
                    warp[0, 2] = float(seed[0])
                    warp[1, 2] = float(seed[1])

                _, warp = cv2.findTransformECC(
                    new_norm,
                    old_norm,
                    warp,
                    motion,
                    criteria,
                    None,
                    5,
                )

                warp_to_apply = cv2.invertAffineTransform(warp)
                dx = float(warp_to_apply[0, 2])
                dy = float(warp_to_apply[1, 2])
                if abs(dx) > max_shift or abs(dy) > max_shift:
                    continue

                height, width = old_image.shape[:2]
                aligned = cv2.warpAffine(
                    old_image,
                    warp_to_apply,
                    (width, height),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )

                before, after = _alignment_quality(new_image, old_image, aligned, dx, dy)
                if not _is_quality_improved(before, after, dx=dx, dy=dy):
                    last_error = (
                        f"ECC 质量不足: before={before:.4f}, after={after:.4f}, motion={motion}"
                    )
                    continue

                rotation = float(np.degrees(np.arctan2(warp_to_apply[1, 0], warp_to_apply[0, 0])))
                result = AlignResult(
                    aligned_old=aligned,
                    dx=dx,
                    dy=dy,
                    rotation=rotation,
                    success=True,
                )
                improvement = after - before
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_result = result
            except cv2.error as exc:
                last_error = f"ECC 收敛失败: {exc}"
                continue

    if best_result is not None:
        return best_result

    return AlignResult(aligned_old=None, success=False, error_message=last_error)


def _find_siril_executable() -> Optional[str]:
    """Find Siril CLI in PATH."""
    for name in ("siril-cli", "siril-cli.exe", "siril", "siril.exe"):
        executable = shutil.which(name)
        if executable:
            logger.info("Siril executable found: %s", executable)
            return executable
    logger.warning("Siril executable not found in PATH")
    return None


def _align_siril(
    new_image: np.ndarray,
    old_image: np.ndarray,
    max_shift: int,
) -> AlignResult:
    """Align with Siril CLI."""

    def _safe_decode(data: bytes) -> str:
        for encoding in ("utf-8", "gbk", "mbcs"):
            try:
                return data.decode(encoding)
            except Exception:
                continue
        return data.decode("utf-8", errors="replace")

    logger.info("Siril alignment start")
    executable = _find_siril_executable()
    if not executable:
        return AlignResult(
            aligned_old=None,
            success=False,
            error_message="未找到 Siril CLI (siril-cli/siril)",
        )

    from astropy.io import fits as astropy_fits

    with tempfile.TemporaryDirectory(prefix="scann_siril_align_") as temp_dir:
        work_dir = Path(temp_dir)

        ref_path = work_dir / "a_ref.fit"
        old_path = work_dir / "b_old.fit"
        script_path = work_dir / "align.ssf"

        new_sanitized = np.nan_to_num(new_image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        old_sanitized = np.nan_to_num(old_image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

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
                    except Exception:
                        pass

            cache_dir = work_dir / "cache"
            if cache_dir.exists() and cache_dir.is_dir():
                try:
                    shutil.rmtree(cache_dir, ignore_errors=True)
                except Exception:
                    pass

            script = "\n".join([
                "requires 1.2.0",
                f'cd "{work_dir.as_posix()}"',
                f"setfindstar reset -sigma={sigma} -roundness={roundness} -relax={relax}",
                "link pair",
                f"setref pair_ {setref_idx}",
                f"register pair_ -transf={transf} -interp=lanczos4 -prefix=r_ -maxstars=2000",
                "exit",
                "",
            ])
            script_path.write_text(script, encoding="utf-8")

            try:
                logger.info(
                    "Running Siril script: %s (setref=%s, transf=%s, sigma=%s, roundness=%s, relax=%s)",
                    script_path,
                    setref_idx,
                    transf,
                    sigma,
                    roundness,
                    relax,
                )
                proc = subprocess.run(
                    [executable, "-d", str(work_dir), "-s", str(script_path)],
                    capture_output=True,
                    text=False,
                    timeout=120,
                    check=False,
                )
                last_proc = proc
                logger.info("Siril finished with rc=%s", proc.returncode)
            except Exception as exc:
                logger.exception("Siril execution failed")
                return AlignResult(
                    aligned_old=None,
                    success=False,
                    error_message=f"调用 Siril 失败: {exc}",
                )

            preferred = [
                work_dir / "r_pair_00002.fit",
                work_dir / "r_pair_00002.fits",
                work_dir / "r_pair_00002.fts",
                work_dir / "R_PAIR_00002.FIT",
                work_dir / "R_PAIR_00002.FITS",
                work_dir / "R_PAIR_00002.FTS",
            ]
            found = next((path for path in preferred if path.is_file()), None)
            if found is not None:
                aligned_old_path = found
                logger.info(
                    "Siril alignment output found: %s (setref=%s, transf=%s)",
                    aligned_old_path,
                    setref_idx,
                    transf,
                )
                break

            out = _safe_decode(proc.stdout or b"")
            err = _safe_decode(proc.stderr or b"")
            logger.warning(
                "Siril attempt failed (setref=%s, transf=%s, sigma=%s, roundness=%s, relax=%s, rc=%s): %s | %s",
                setref_idx,
                transf,
                sigma,
                roundness,
                relax,
                proc.returncode,
                out[-200:],
                err[-200:],
            )

        if aligned_old_path is None:
            proc = last_proc
            out = _safe_decode((proc.stdout if proc else b"") or b"")
            err = _safe_decode((proc.stderr if proc else b"") or b"")
            produced = ", ".join(sorted(path.name for path in work_dir.iterdir()))
            logger.warning("Siril did not produce aligned output. rc=%s", (proc.returncode if proc else "N/A"))
            return AlignResult(
                aligned_old=None,
                success=False,
                error_message=(
                    f"Siril 未生成对齐结果: rc={(proc.returncode if proc else 'N/A')}; "
                    f"out={out[-500:]}; err={err[-500:]}; files={produced}"
                ),
            )

        aligned: Optional[np.ndarray] = None
        aligned_path_str = str(aligned_old_path.resolve())
        last_exc: Optional[Exception] = None
        for _ in range(10):
            try:
                if not Path(aligned_path_str).is_file():
                    time.sleep(0.2)
                    continue
                with astropy_fits.open(aligned_path_str, memmap=False) as hdul:
                    data = hdul[0].data
                if data is None:
                    last_exc = RuntimeError("Siril 输出为空图像")
                    time.sleep(0.2)
                    continue
                aligned = np.array(data, copy=True)
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(0.2)

        if aligned is None:
            logger.exception("Read Siril output failed")
            return AlignResult(
                aligned_old=None,
                success=False,
                error_message=f"读取 Siril 结果失败: {last_exc}",
            )

        aligned = _match_intensity_scale(aligned, old_image)

        estimated_shift = _estimate_translation(new_image, aligned)
        if estimated_shift is None:
            return AlignResult(
                aligned_old=None,
                success=False,
                error_message="Siril 结果无法估计相对新图偏移",
            )

        dx, dy = estimated_shift
        logger.info("Siril estimated shift: dx=%.3f dy=%.3f", dx, dy)
        if abs(dx) > max_shift or abs(dy) > max_shift:
            return AlignResult(
                aligned_old=None,
                dx=dx,
                dy=dy,
                success=False,
                error_message=f"Siril 偏移量过大: dx={dx:.3f}, dy={dy:.3f}",
            )

        before, after = _alignment_quality(new_image, old_image, aligned, dx, dy)
        if not _is_quality_improved(before, after, dx=dx, dy=dy):
            logger.warning(
                "Siril 注册结果未通过本地质量门，但优先信任 Siril 输出: "
                "before=%.4f, after=%.4f, dx=%.3f, dy=%.3f",
                before,
                after,
                dx,
                dy,
            )

        return AlignResult(
            aligned_old=aligned,
            dx=dx,
            dy=dy,
            success=True,
        )


def _align_feature_matching(
    new_image: np.ndarray,
    old_image: np.ndarray,
    max_shift: int,
) -> AlignResult:
    """Affine alignment using feature matching."""
    import cv2

    def _prepare_feature_image(image: np.ndarray) -> np.ndarray:
        enhanced = _enhance_stars(_normalize_for_alignment(_to_gray_f32(image)))
        return cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    new_gray = _prepare_feature_image(new_image)
    old_gray = _prepare_feature_image(old_image)

    detectors = [
        ("AKAZE", cv2.AKAZE_create()),
        ("ORB", cv2.ORB_create(nfeatures=4000, edgeThreshold=5, fastThreshold=5)),
    ]

    best_result: AlignResult | None = None
    best_improvement = float("-inf")
    last_error = "特征匹配失败"

    for detector_name, detector in detectors:
        kp1, des1 = detector.detectAndCompute(new_gray, None)
        kp2, des2 = detector.detectAndCompute(old_gray, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            last_error = f"{detector_name} 特征点不足"
            continue

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        matches = [
            first for first, second in raw_matches
            if second is not None and first.distance < 0.75 * second.distance
        ]
        matches = sorted(matches, key=lambda match: match.distance)[:200]
        if len(matches) < 8:
            last_error = f"{detector_name} 有效匹配点不足: {len(matches)}"
            continue

        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        matrix, mask = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=5000,
            confidence=0.99,
        )
        if matrix is None or mask is None:
            last_error = f"{detector_name} 无法估计仿射变换"
            continue

        inliers = int(np.sum(mask))
        if inliers < 8 or inliers < max(8, int(len(matches) * 0.25)):
            last_error = f"{detector_name} RANSAC 内点不足: {inliers}/{len(matches)}"
            continue

        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        if not 0.8 <= scale <= 1.2:
            last_error = f"{detector_name} 尺度异常: {scale:.3f}"
            continue

        dx = float(matrix[0, 2])
        dy = float(matrix[1, 2])
        if abs(dx) > max_shift or abs(dy) > max_shift:
            last_error = f"{detector_name} 偏移量过大: dx={dx:.1f}, dy={dy:.1f}"
            continue

        height, width = old_image.shape[:2]
        aligned = cv2.warpAffine(
            old_image,
            matrix,
            (width, height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        before, after = _alignment_quality(new_image, old_image, aligned, dx, dy)
        if not _is_quality_improved(before, after, dx=dx, dy=dy):
            last_error = (
                f"{detector_name} 质量不足: before={before:.4f}, after={after:.4f}, "
                f"inliers={inliers}/{len(matches)}"
            )
            continue

        rotation = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
        result = AlignResult(
            aligned_old=aligned,
            dx=dx,
            dy=dy,
            rotation=rotation,
            success=True,
        )
        improvement = after - before
        if improvement > best_improvement:
            best_improvement = improvement
            best_result = result

    if best_result is not None:
        return best_result

    return AlignResult(aligned_old=None, success=False, error_message=last_error)


def batch_align(
    new_images: List[np.ndarray],
    old_images: List[np.ndarray],
    method: str = "auto",
    max_shift: int = 100,
) -> List[AlignResult]:
    """Batch-align image pairs."""
    if len(new_images) != len(old_images):
        raise ValueError("新旧图列表长度不一致")

    results = []
    for new_img, old_img in zip(new_images, old_images):
        results.append(align(new_img, old_img, method=method, max_shift=max_shift))
    return results
