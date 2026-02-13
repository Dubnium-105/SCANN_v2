"""图像对齐模块

职责:
- 以新图为参考图，仅移动旧图进行对齐
- 绝不移动新图！
- 支持批量对齐
"""

from __future__ import annotations

from typing import List, Optional
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from scann.core.models import AlignResult, FitsImage


logger = logging.getLogger(__name__)


def align(
    new_image: np.ndarray,
    old_image: np.ndarray,
    method: str = "phase_correlation",
    max_shift: int = 100,
) -> AlignResult:
    """对齐旧图到新图

    以新图作为参考图，绝不移动新图，只移动旧图。

    Args:
        new_image: 新图像素数据 (参考图，不可移动)
        old_image: 旧图像素数据 (需要对齐的图)
        method: 对齐算法 ("phase_correlation"/"auto", "siril", "ecc", "feature_matching")
        max_shift: 最大允许偏移量 (像素)

    Returns:
        AlignResult: 对齐结果，包含对齐后的旧图和偏移参数
    """
    if new_image.shape != old_image.shape:
        return AlignResult(
            aligned_old=None,
            success=False,
            error_message=f"图像尺寸不匹配: new={new_image.shape}, old={old_image.shape}",
        )

    try:
        if method in {"phase_correlation", "auto"}:
            # 先尝试稳健的相位相关；失败后自动回退到 ECC/特征匹配
            result = _align_phase_correlation(new_image, old_image, max_shift)
            if result.success:
                return result

            ecc_result = _align_ecc(new_image, old_image, max_shift)
            if ecc_result.success:
                return ecc_result

            feature_result = _align_feature_matching(new_image, old_image, max_shift)
            if feature_result.success:
                return feature_result

            return AlignResult(
                aligned_old=None,
                success=False,
                error_message=(
                    f"phase失败: {result.error_message}; "
                    f"ECC失败: {ecc_result.error_message}; "
                    f"feature失败: {feature_result.error_message}"
                ),
            )

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
    except Exception as e:
        return AlignResult(
            aligned_old=None,
            success=False,
            error_message=str(e),
        )


def _to_gray_f32(image: np.ndarray) -> np.ndarray:
    """转灰度 float32，并清理 NaN/Inf。"""
    import cv2

    if image.ndim == 3:
        gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.float32)
    return np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_for_alignment(gray_f32: np.ndarray) -> np.ndarray:
    """鲁棒归一化到 [0, 1]，减弱背景与亮度尺度差异。"""
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
    """星点增强（高通），更接近 v1 常用的“去背景后配准”思路。"""
    import cv2

    low = cv2.GaussianBlur(norm01, (0, 0), sigmaX=2.0)
    high = norm01 - low
    high = np.clip(high, 0.0, None)
    m = float(np.max(high))
    if m > 0:
        high /= m
    return high.astype(np.float32)


def _warp_translate(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    import cv2

    h, w = image.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _zncc(a: np.ndarray, b: np.ndarray) -> float:
    """零均值归一化互相关，范围约 [-1, 1]。"""
    aa = a.astype(np.float32).ravel()
    bb = b.astype(np.float32).ravel()
    am = float(np.mean(aa))
    bm = float(np.mean(bb))
    aa -= am
    bb -= bm
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return -1.0
    return float(np.dot(aa, bb) / denom)


def _match_intensity_scale(aligned: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """将对齐结果亮度范围匹配到参考图（用于 Siril 输出归一化场景）。"""
    a = np.nan_to_num(aligned.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    r = np.nan_to_num(reference.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    a1, a99 = np.percentile(a, [1, 99])
    r1, r99 = np.percentile(r, [1, 99])
    ar = float(a99 - a1)
    rr = float(r99 - r1)

    # Siril 常见输出：32-bit 归一化到 [0,1]，导致回写后画面接近纯黑
    # 仅在范围明显不一致时做线性匹配，避免影响正常情况。
    if ar <= 1e-6:
        return a

    ratio = rr / ar if ar > 0 else 1.0
    if ratio < 20.0 and ratio > 0.05:
        return a

    mapped = (a - a1) * (rr / ar) + r1
    # 使用参考图极值裁剪，防止异常值污染后续显示/处理
    rmin = float(np.min(r))
    rmax = float(np.max(r))
    mapped = np.clip(mapped, rmin, rmax)
    return mapped.astype(np.float32)


def _align_phase_correlation(
    new_image: np.ndarray,
    old_image: np.ndarray,
    max_shift: int,
) -> AlignResult:
    """稳健相位相关法对齐（多尺度 + 星点增强 + 质量验证）。"""
    import cv2

    new_g = _to_gray_f32(new_image)
    old_g = _to_gray_f32(old_image)
    new_n = _enhance_stars(_normalize_for_alignment(new_g))
    old_n = _enhance_stars(_normalize_for_alignment(old_g))

    # 多尺度从粗到细
    scales = [0.25, 0.5, 1.0]
    total_dx = 0.0
    total_dy = 0.0
    last_response = 0.0

    for s in scales:
        h, w = new_n.shape[:2]
        ws = max(32, int(round(w * s)))
        hs = max(32, int(round(h * s)))

        new_s = cv2.resize(new_n, (ws, hs), interpolation=cv2.INTER_AREA)
        old_s = cv2.resize(old_n, (ws, hs), interpolation=cv2.INTER_AREA)

        # 先按上层结果预平移，再估计残差
        preshift_dx = total_dx * s
        preshift_dy = total_dy * s
        old_s_pre = _warp_translate(old_s, preshift_dx, preshift_dy)

        window = cv2.createHanningWindow((ws, hs), cv2.CV_32F)
        (ddx, ddy), response = cv2.phaseCorrelate(new_s, old_s_pre, window)
        last_response = float(response)

        # 反投影到全分辨率
        total_dx += float(ddx) / s
        total_dy += float(ddy) / s

    if abs(total_dx) > max_shift or abs(total_dy) > max_shift:
        return AlignResult(
            aligned_old=None,
            dx=total_dx,
            dy=total_dy,
            success=False,
            error_message=(
                f"偏移量过大: dx={total_dx:.1f}, dy={total_dy:.1f} "
                f"(max={max_shift})"
            ),
        )

    aligned = _warp_translate(old_image, total_dx, total_dy)

    # 质量验证：对齐后相关性应明显变好
    before = _zncc(new_n, old_n)
    aligned_n = _enhance_stars(_normalize_for_alignment(_to_gray_f32(aligned)))
    after = _zncc(new_n, aligned_n)
    if after < before + 0.01:
        return AlignResult(
            aligned_old=None,
            dx=total_dx,
            dy=total_dy,
            success=False,
            error_message=(
                f"相位相关质量不足: before={before:.4f}, after={after:.4f}, "
                f"response={last_response:.4f}"
            ),
        )

    return AlignResult(aligned_old=aligned, dx=total_dx, dy=total_dy, success=True)


def _align_ecc(
    new_image: np.ndarray,
    old_image: np.ndarray,
    max_shift: int,
) -> AlignResult:
    """ECC 配准兜底（先平移，再欧氏）。"""
    import cv2

    new_n = _normalize_for_alignment(_to_gray_f32(new_image))
    old_n = _normalize_for_alignment(_to_gray_f32(old_image))

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        200,
        1e-6,
    )

    for motion in (cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN):
        try:
            warp = np.eye(2, 3, dtype=np.float32)
            _, warp = cv2.findTransformECC(
                new_n,
                old_n,
                warp,
                motion,
                criteria,
                None,
                5,
            )

            dx = float(warp[0, 2])
            dy = float(warp[1, 2])
            if abs(dx) > max_shift or abs(dy) > max_shift:
                continue

            h, w = old_image.shape[:2]
            aligned = cv2.warpAffine(
                old_image,
                warp,
                (w, h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            rotation = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
            return AlignResult(
                aligned_old=aligned,
                dx=dx,
                dy=dy,
                rotation=rotation,
                success=True,
            )
        except cv2.error:
            continue

    return AlignResult(
        aligned_old=None,
        success=False,
        error_message="ECC 收敛失败",
    )


def _find_siril_executable() -> Optional[str]:
    """查找 Siril CLI 可执行文件。"""
    for name in ("siril-cli", "siril-cli.exe", "siril", "siril.exe"):
        exe = shutil.which(name)
        if exe:
            logger.info("Siril executable found: %s", exe)
            return exe
    logger.warning("Siril executable not found in PATH")
    return None


def _align_siril(
    new_image: np.ndarray,
    old_image: np.ndarray,
    max_shift: int,
) -> AlignResult:
    """使用 Siril CLI 对齐（星点配准）。"""
    def _safe_decode(data: bytes) -> str:
        """兼容 Windows 本地编码与 UTF-8 的稳健解码。"""
        for enc in ("utf-8", "gbk", "mbcs"):
            try:
                return data.decode(enc)
            except Exception:
                continue
        return data.decode("utf-8", errors="replace")

    logger.info("Siril alignment start")
    exe = _find_siril_executable()
    if not exe:
        return AlignResult(
            aligned_old=None,
            success=False,
            error_message="未找到 Siril CLI (siril-cli/siril)",
        )

    # 延迟导入，避免无关路径依赖
    from astropy.io import fits as astropy_fits

    with tempfile.TemporaryDirectory(prefix="scann_siril_align_") as td:
        work = Path(td)

        ref_path = work / "a_ref.fit"
        old_path = work / "b_old.fit"
        script_path = work / "align.ssf"

        # Siril 对输入数据比较敏感：先清理 NaN/Inf 并统一为 float32
        new_sanitized = np.nan_to_num(new_image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        old_sanitized = np.nan_to_num(old_image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        astropy_fits.PrimaryHDU(data=new_sanitized).writeto(ref_path, overwrite=True)
        astropy_fits.PrimaryHDU(data=old_sanitized).writeto(old_path, overwrite=True)

        # Siril 在不同版本上对 setref 索引与变换模型容错差异较大，这里做多策略尝试。
        # 逐步放宽星点检测条件，提升稀疏星场/低信噪场景下的成功率
        attempts = [
            ("1", "affine", "2.5", "0.25", "off"),
            ("1", "similarity", "1.6", "0.15", "on"),
            ("1", "shift", "1.2", "0.10", "on"),
        ]

        aligned_old_path = None
        last_proc = None

        for setref_idx, transf, sigma, roundness, relax in attempts:
            # 清理 Siril 生成工件，避免多次 link/register 互相污染
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
            for pat in cleanup_patterns:
                for p in work.glob(pat):
                    try:
                        p.unlink()
                    except Exception:
                        pass

            cache_dir = work / "cache"
            if cache_dir.exists() and cache_dir.is_dir():
                try:
                    shutil.rmtree(cache_dir, ignore_errors=True)
                except Exception:
                    pass

            script = "\n".join([
                "requires 1.2.0",
                f'cd "{work.as_posix()}"',
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
                    [exe, "-d", str(work), "-s", str(script_path)],
                    capture_output=True,
                    text=False,
                    timeout=120,
                    check=False,
                )
                last_proc = proc
                logger.info("Siril finished with rc=%s", proc.returncode)
            except Exception as e:
                logger.exception("Siril execution failed")
                return AlignResult(
                    aligned_old=None,
                    success=False,
                    error_message=f"调用 Siril 失败: {e}",
                )

            # 期望第二帧是 old 的对齐结果 (00002)
            preferred = [
                work / "r_pair_00002.fit",
                work / "r_pair_00002.fits",
                work / "r_pair_00002.fts",
                work / "R_PAIR_00002.FIT",
                work / "R_PAIR_00002.FITS",
                work / "R_PAIR_00002.FTS",
            ]
            found = next((p for p in preferred if p.is_file()), None)
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
            tail = out[-500:]
            err_tail = err[-500:]
            produced = ", ".join(sorted([p.name for p in work.iterdir()]))
            logger.warning("Siril did not produce aligned output. rc=%s", (proc.returncode if proc else "N/A"))
            return AlignResult(
                aligned_old=None,
                success=False,
                error_message=(
                    f"Siril 未生成对齐结果: rc={(proc.returncode if proc else 'N/A')}; "
                    f"out={tail}; err={err_tail}; files={produced}"
                ),
            )

        # Windows 下偶发 Siril 刚落盘即读导致异常，做短暂重试
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
            except Exception as e:
                last_exc = e
                time.sleep(0.2)

        if aligned is None:
            logger.exception("Read Siril output failed")
            return AlignResult(
                aligned_old=None,
                success=False,
                error_message=f"读取 Siril 结果失败: {last_exc}",
            )

        # Siril 结果在部分版本中为归一化 32-bit，这里将其映射回旧图亮度范围
        aligned = _match_intensity_scale(aligned, old_image)

        # 用相位相关估计最终平移量，便于统一输出 dx/dy
        phase = _align_phase_correlation(new_image, aligned, max_shift=max_shift)
        dx = float(phase.dx) if phase.success else 0.0
        dy = float(phase.dy) if phase.success else 0.0

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
    """特征点匹配法对齐 (适用于旋转+平移)"""
    import cv2

    # 转灰度
    if new_image.ndim == 3:
        new_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    else:
        new_gray = new_image.copy()

    if old_image.ndim == 3:
        old_gray = cv2.cvtColor(old_image, cv2.COLOR_BGR2GRAY)
    else:
        old_gray = old_image.copy()

    # 确保 uint8
    if new_gray.dtype != np.uint8:
        new_gray = cv2.normalize(new_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if old_gray.dtype != np.uint8:
        old_gray = cv2.normalize(old_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # ORB 特征检测
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(new_gray, None)
    kp2, des2 = orb.detectAndCompute(old_gray, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return AlignResult(
            aligned_old=None,
            success=False,
            error_message="特征点不足，无法对齐",
        )

    # 匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 10:
        return AlignResult(
            aligned_old=None,
            success=False,
            error_message=f"匹配点不足: {len(matches)}",
        )

    # 提取匹配点
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 估算变换矩阵 (仿射或刚体)
    M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
    if M is None:
        return AlignResult(
            aligned_old=None,
            success=False,
            error_message="无法估算变换矩阵",
        )

    dx = float(M[0, 2])
    dy = float(M[1, 2])
    if abs(dx) > max_shift or abs(dy) > max_shift:
        return AlignResult(
            aligned_old=None,
            dx=dx,
            dy=dy,
            success=False,
            error_message=f"特征匹配偏移量过大: dx={dx:.1f}, dy={dy:.1f}",
        )
    # 从仿射矩阵提取旋转角
    rotation = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))

    h, w = old_image.shape[:2]
    aligned = cv2.warpAffine(
        old_image, M, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return AlignResult(
        aligned_old=aligned,
        dx=dx,
        dy=dy,
        rotation=rotation,
        success=True,
    )


def batch_align(
    new_images: List[np.ndarray],
    old_images: List[np.ndarray],
    method: str = "phase_correlation",
    max_shift: int = 100,
) -> List[AlignResult]:
    """批量对齐

    Args:
        new_images: 新图列表 (参考图)
        old_images: 旧图列表 (待对齐)
        method: 对齐方法
        max_shift: 最大允许偏移量

    Returns:
        对齐结果列表
    """
    if len(new_images) != len(old_images):
        raise ValueError("新旧图列表长度不一致")

    results = []
    for new_img, old_img in zip(new_images, old_images):
        result = align(new_img, old_img, method=method, max_shift=max_shift)
        results.append(result)

    return results
