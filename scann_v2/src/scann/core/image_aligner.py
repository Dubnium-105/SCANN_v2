"""图像对齐模块

职责:
- 以新图为参考图，仅移动旧图进行对齐
- 绝不移动新图！
- 支持批量对齐
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from scann.core.models import AlignResult, FitsImage


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
        method: 对齐算法 ("phase_correlation", "feature_matching")
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
        if method == "phase_correlation":
            return _align_phase_correlation(new_image, old_image, max_shift)
        elif method == "feature_matching":
            return _align_feature_matching(new_image, old_image, max_shift)
        else:
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


def _align_phase_correlation(
    new_image: np.ndarray,
    old_image: np.ndarray,
    max_shift: int,
) -> AlignResult:
    """相位相关法对齐"""
    import cv2

    # 转为浮点
    new_f = new_image.astype(np.float32)
    old_f = old_image.astype(np.float32)

    # 如果是多通道，转灰度
    if new_f.ndim == 3:
        new_f = cv2.cvtColor(new_f, cv2.COLOR_BGR2GRAY)
    if old_f.ndim == 3:
        old_f = cv2.cvtColor(old_f, cv2.COLOR_BGR2GRAY)

    # 相位相关
    (dx, dy), response = cv2.phaseCorrelate(new_f, old_f)

    # 检查偏移量是否合理
    if abs(dx) > max_shift or abs(dy) > max_shift:
        return AlignResult(
            aligned_old=None,
            dx=dx,
            dy=dy,
            success=False,
            error_message=f"偏移量过大: dx={dx:.1f}, dy={dy:.1f} (max={max_shift})",
        )

    # 构造平移矩阵并移动旧图
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = old_image.shape[:2]
    aligned = cv2.warpAffine(
        old_image,
        M,
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return AlignResult(aligned_old=aligned, dx=dx, dy=dy, success=True)


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
