"""检测流程控制器。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from scann.core.candidate_detector import DetectionParams
from scann.core.fits_io import read_fits, write_fits
from scann.core.image_aligner import align
from scann.core.image_processor import denoise, pseudo_flat_field
from scann.core.models import TargetVerdict
from scann.data.file_manager import scan_fits_folder
from scann.services.detection_service import DetectionPipeline
from scann.services.exclusion_service import ExclusionService

if TYPE_CHECKING:
    from scann.gui.main_window import MainWindow


class DetectionController:
    """集中主窗口中的检测相关事件入口。"""

    def __init__(self, window: MainWindow) -> None:
        self._window = window
        self._exclusion_service: ExclusionService | None = None
        self._exclusion_service_mpcorb_path: str = ""

    def _get_exclusion_service(self) -> ExclusionService | None:
        config = getattr(self._window, "_config", None)
        if config is None:
            if self._exclusion_service is None:
                self._exclusion_service = ExclusionService()
            return self._exclusion_service

        mpcorb_path = getattr(config, "mpcorb_path", "") or ""
        if self._exclusion_service is not None and self._exclusion_service_mpcorb_path == mpcorb_path:
            return self._exclusion_service

        service = ExclusionService(
            mpcorb_path=mpcorb_path or None,
            observatory=getattr(config, "observatory", None),
        )
        if mpcorb_path:
            service.load_mpcorb()
        self._exclusion_service = service
        self._exclusion_service_mpcorb_path = mpcorb_path
        return service

    def get_exclusion_service(self) -> ExclusionService:
        return self._get_exclusion_service()

    def _resolve_current_new_image_path(self) -> str | None:
        index = getattr(self._window, "_current_pair_idx", -1)
        image_pairs = getattr(self._window, "_image_pairs", [])
        if index < 0 or index >= len(image_pairs):
            return None

        pair = image_pairs[index]
        resolver = getattr(self._window, "_resolve_pair_image_paths", None)
        if callable(resolver):
            new_path, _old_path, _using_aligned = resolver(pair)
            return str(new_path)

        new_path = getattr(pair, "new_path", None)
        return str(new_path) if new_path is not None else None

    def resolve_current_new_image_path(self) -> str | None:
        return self._resolve_current_new_image_path()

    def _align_marked_to_new(
        self,
        new_data: np.ndarray,
        marked_data: np.ndarray,
        *,
        pair_name: str,
    ) -> np.ndarray:
        aligned_marked = marked_data
        if new_data.shape[:2] != marked_data.shape[:2]:
            self._window._logger.warning(
                "带标记新图与新图尺寸不一致，无法执行额外对齐: %s; new=%s marked=%s",
                pair_name,
                new_data.shape,
                marked_data.shape,
            )
            return aligned_marked

        max_shift = max(100, int(min(new_data.shape[:2]) * 0.45))
        marked_result = align(
            new_data,
            marked_data,
            method="siril",
            max_shift=max_shift,
        )
        if float(getattr(marked_result, "rotation", 0.0) or 0.0) == 180.0:
            self._window._logger.info(
                "Marked image aligned after Siril 180-degree rotation: %s",
                pair_name,
            )
        if marked_result.success and marked_result.aligned_old is not None:
            return np.nan_to_num(
                marked_result.aligned_old.astype(np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        self._window._logger.warning(
            "带标记新图对齐新图失败，旋转180度兜底后仍未成功: %s (%s)",
            pair_name,
            marked_result.error_message,
        )
        return aligned_marked

    def batch_align(self, force: bool = False) -> None:
        """执行批量对齐。
        
        Args:
            force: 是否强制重新对齐（忽略已有的对齐产物）
        """
        if not self._window._image_pairs:
            self._window._show_message("请先加载新旧图文件夹配对")
            return

        success_count = 0
        fail_count = 0
        skip_count = 0
        total = len(self._window._image_pairs)

        self._window.progress_bar.setVisible(True)
        self._window.progress_bar.setRange(0, total)
        self._window.progress_bar.setValue(0)
        self._window.btn_align.setEnabled(False)
        self._window.act_align.setEnabled(False)

        # 获取键盘修饰键状态
        modifiers = QApplication.keyboardModifiers()
        force_realign = force or (modifiers & Qt.ShiftModifier)

        if force_realign:
            self._window._show_message(f"强制重新对齐模式: 共 {total} 对", 3000)
        else:
            self._window._show_message(f"开始批量对齐: 共 {total} 对", 3000)

        try:
            for idx, pair in enumerate(self._window._image_pairs, start=1):
                try:
                    if not force_realign and self._window._pair_has_aligned_artifacts(pair):
                        skip_count += 1
                        self._window._logger.info(
                            "[%s/%s] 已有对齐裁剪标记，跳过: %s",
                            idx,
                            total,
                            pair.name,
                        )
                        self._window._show_message(
                            f"[{idx}/{total}] 跳过已对齐: {pair.name}",
                            1000,
                        )
                        self._window.progress_bar.setValue(idx)
                        QApplication.processEvents()
                        continue

                    new_fits = read_fits(pair.new_path)
                    old_fits = read_fits(pair.old_path)

                    new_data = np.nan_to_num(
                        new_fits.data.astype(np.float32),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    old_data = np.nan_to_num(
                        old_fits.data.astype(np.float32),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    new_span = float(
                        np.percentile(new_data, 99.5) - np.percentile(new_data, 0.5)
                    )
                    old_span = float(
                        np.percentile(old_data, 99.5) - np.percentile(old_data, 0.5)
                    )
                    if new_span <= 1e-6 or old_span <= 1e-6:
                        fail_count += 1
                        self._window._logger.error(
                            "[%s/%s] 对齐前校验失败: %s; new_span=%.6g, old_span=%.6g; new=%s old=%s",
                            idx,
                            total,
                            pair.name,
                            new_span,
                            old_span,
                            pair.new_path,
                            pair.old_path,
                        )
                        self._window._show_message(
                            f"[{idx}/{total}] 跳过无效图像: {pair.name} (新图/旧图近乎常量)",
                            2000,
                            level="WARNING",
                        )
                        self._window.progress_bar.setValue(idx)
                        QApplication.processEvents()
                        continue

                    self._window._logger.info(
                        "[%s/%s] 开始 Siril 对齐: %s",
                        idx,
                        total,
                        pair.name,
                    )
                    self._window._show_message(
                        f"[{idx}/{total}] Siril 对齐: {pair.name}",
                        1000,
                    )

                    result = align(new_fits.data, old_fits.data, method="siril")
                    if not result.success or result.aligned_old is None:
                        self._window._logger.warning(
                            "[%s/%s] Siril alignment failed: %s; reason=%s",
                            idx,
                            total,
                            pair.name,
                            result.error_message,
                        )
                        self._window._show_message(
                            f"[{idx}/{total}] Siril alignment failed: {pair.name}",
                            1500,
                            level="WARNING",
                        )

                    if result.success and result.aligned_old is not None:
                        h, w = new_data.shape[:2]
                        crop_bounds = self._window._calc_overlap_crop_bounds(
                            w=w,
                            h=h,
                            dx=result.dx,
                            dy=result.dy,
                            aligned_old=result.aligned_old,
                            new_image=new_data,
                        )
                        if crop_bounds is None:
                            fail_count += 1
                            self._window._logger.error(
                                "[%s/%s] 对齐后无有效重叠区域: %s; dx=%.3f dy=%.3f",
                                idx,
                                total,
                                pair.name,
                                result.dx,
                                result.dy,
                            )
                            self._window._show_message(
                                f"[{idx}/{total}] 对齐失败(无重叠区域): {pair.name}",
                                2000,
                                level="WARNING",
                            )
                            self._window.progress_bar.setValue(idx)
                            QApplication.processEvents()
                            continue

                        x0, x1, y0, y1 = crop_bounds
                        cropped_new = new_data[y0:y1, x0:x1]
                        cropped_old = result.aligned_old[y0:y1, x0:x1]

                        (
                            new_aligned_path,
                            old_aligned_path,
                            new_marker_path,
                            old_marker_path,
                        ) = self._window._aligned_artifact_paths(pair)
                        write_fits(new_aligned_path, cropped_new, new_fits.header)
                        write_fits(old_aligned_path, cropped_old, old_fits.header)
                        marker_text = (
                            "aligned=1\n"
                            "method=siril\n"
                            f"rotation={float(getattr(result, 'rotation', 0.0) or 0.0):.0f}\n"
                            f"dx={result.dx:.6f}\n"
                            f"dy={result.dy:.6f}\n"
                            f"crop={x0},{x1},{y0},{y1}\n"
                        )
                        new_marker_path.write_text(marker_text, encoding="utf-8")
                        old_marker_path.write_text(marker_text, encoding="utf-8")
                        pair_service = getattr(self._window, "pair_service", None)
                        marked_source_path = (
                            pair_service.resolve_marked_image_path(pair.new_path)
                            if pair_service is not None
                            else None
                        )
                        marked_aligned_path = (
                            pair_service.derive_marked_image_path(new_aligned_path)
                            if pair_service is not None
                            else None
                        )
                        if marked_source_path is not None and marked_aligned_path is not None:
                            try:
                                marked_fits = read_fits(marked_source_path)
                                marked_data = np.nan_to_num(
                                    marked_fits.data.astype(np.float32),
                                    nan=0.0,
                                    posinf=0.0,
                                    neginf=0.0,
                                )
                                marked_data = self._align_marked_to_new(
                                    new_data,
                                    marked_data,
                                    pair_name=pair.name,
                                )
                                if marked_data.shape[0] >= y1 and marked_data.shape[1] >= x1:
                                    cropped_marked = marked_data[y0:y1, x0:x1]
                                    write_fits(marked_aligned_path, cropped_marked, marked_fits.header)
                                else:
                                    self._window._logger.warning(
                                        "[%s/%s] 带标记新图尺寸不足，无法复用新图裁剪: %s; marked=%s crop=(%s,%s,%s,%s)",
                                        idx,
                                        total,
                                        pair.name,
                                        marked_data.shape,
                                        x0,
                                        x1,
                                        y0,
                                        y1,
                                    )
                            except Exception as marked_exc:
                                self._window._logger.warning(
                                    "[%s/%s] 带标记新图裁剪保存失败: %s (%s)",
                                    idx,
                                    total,
                                    pair.name,
                                    marked_exc,
                                )

                        success_count += 1
                        self._window._logger.info(
                            "[%s/%s] 对齐成功并保存裁剪重叠图: %s; new=%s old=%s",
                            idx,
                            total,
                            pair.name,
                            new_aligned_path,
                            old_aligned_path,
                        )
                    else:
                        fail_count += 1
                        self._window._logger.error(
                            "[%s/%s] 对齐失败: %s; reason=%s",
                            idx,
                            total,
                            pair.name,
                            result.error_message,
                        )
                except Exception as exc:
                    fail_count += 1
                    self._window._logger.exception(
                        "[%s/%s] 对齐异常: %s",
                        idx,
                        total,
                        pair.name,
                    )
                    self._window._show_message(
                        f"[{idx}/{total}] 对齐异常: {pair.name} ({exc})",
                        2000,
                        level="ERROR",
                    )

                self._window.progress_bar.setValue(idx)
                QApplication.processEvents()
        finally:
            self._window.progress_bar.setVisible(False)
            self._window.btn_align.setEnabled(True)
            self._window.act_align.setEnabled(True)

        self._window._show_message(
            f"对齐完成: 成功 {success_count}, 跳过 {skip_count}, 失败 {fail_count}",
            5000,
        )

        if self._window._current_pair_idx >= 0:
            self._window._load_pair(self._window._current_pair_idx)

    def batch_process(self) -> None:
        from scann.gui.dialogs.batch_process_dialog import BatchProcessDialog

        dialog = BatchProcessDialog(self._window)
        dialog.process_started.connect(self.run_batch_process)
        self._window._batch_dialog = dialog
        dialog.exec_()

    def run_batch_process(self, params: dict) -> None:
        input_dir = params.get("input_dir", self._window._new_folder)
        output_dir = params.get("output_dir", "")
        if not input_dir:
            self._window._show_message("未指定输入文件夹")
            return

        if not output_dir:
            output_dir = str(Path(input_dir) / "processed")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        fits_files = scan_fits_folder(input_dir)
        if not fits_files:
            self._window._show_message("输入文件夹中未找到 FITS 文件")
            return

        success_count = 0
        fail_count = 0
        denoise_method_map = {
            "中值滤波": "median",
            "高斯滤波": "gaussian",
            "双边滤波": "bilateral",
        }

        for index, fits_path in enumerate(fits_files):
            try:
                fits_img = read_fits(str(fits_path))
                data = fits_img.data

                if params.get("denoise", False):
                    method = denoise_method_map.get(
                        params.get("denoise_method", "中值滤波"),
                        "median",
                    )
                    kernel = params.get("kernel_size", 3)
                    data = denoise(data, method=method, kernel_size=kernel)

                if params.get("flat_field", False):
                    sigma = params.get("flat_sigma", 100.0)
                    kernel_size = max(3, int(sigma) * 2 + 1)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    data = pseudo_flat_field(data, kernel_size=kernel_size)

                out_path = str(Path(output_dir) / fits_path.name)
                write_fits(data, out_path)
                success_count += 1

                try:
                    if self._window._batch_dialog is not None:
                        self._window._batch_dialog.update_progress(
                            index + 1,
                            len(fits_files),
                            fits_path.name,
                        )
                except (AttributeError, RuntimeError):
                    pass
            except Exception:
                fail_count += 1

        try:
            if self._window._batch_dialog is not None:
                self._window._batch_dialog.processing_finished()
        except (AttributeError, RuntimeError):
            pass

        self._window._show_message(
            f"批量处理完成: 成功 {success_count}, 失败 {fail_count}",
            5000,
        )

    def build_detection_params(self) -> DetectionParams:
        return DetectionParams(
            thresh=self._window._config.thresh,
            min_area=self._window._config.min_area,
            max_area=self._window._config.max_area,
            sharpness_min=self._window._config.sharpness,
            sharpness_max=self._window._config.max_sharpness,
            contrast_min=self._window._config.contrast,
            edge_margin=self._window._config.edge_margin,
            dynamic_thresh=self._window._config.dynamic_thresh,
            kill_flat=self._window._config.kill_flat,
            kill_dipole=self._window._config.kill_dipole,
            aspect_ratio_max=self._window._config.aspect_ratio_max,
            extent_max=self._window._config.extent_max,
            topk=self._window._config.topk,
        )

    def batch_detect(self) -> None:
        if self._window._new_image_data is None:
            self._window._show_message("请先加载图像数据")
            return

        old_data = self._window._old_image_data
        if old_data is None:
            old_data = np.zeros_like(self._window._new_image_data)

        pipeline = DetectionPipeline(
            detection_params=self.build_detection_params(),
            inference_engine=self._window._inference_engine,
            exclusion_service=self._get_exclusion_service(),
            patch_size=getattr(self._window._config, "slice_size", 80),
            detection_mode=getattr(self._window._config, "detection_mode", "patch"),
            hybrid_primary_mode=getattr(
                self._window._config,
                "hybrid_primary_mode",
                "full_image",
            ),
            hybrid_low_confidence=getattr(
                self._window._config,
                "hybrid_low_confidence",
                0.5,
            ),
        )
        result = pipeline.process_pair(
            pair_name="current",
            new_data=self._window._new_image_data,
            old_data=old_data,
            skip_align=bool(
                self._window.__dict__.get("_current_pair_using_aligned", False)
            ),
            header=getattr(self._window, "_new_fits_header", None),
            image_path=self._resolve_current_new_image_path(),
        )

        if result.candidates:
            self._window.set_candidates(result.candidates)
            self._window._candidates_cache[self._window._current_pair_idx] = result.candidates
            self._window._show_message(
                f"检测完成: 发现 {len(result.candidates)} 个候选体",
                5000,
            )
            return

        self._window._show_message(
            f"检测完成: 未发现候选体 {result.error or ''}",
            5000,
        )
        self._window._candidates_cache[self._window._current_pair_idx] = []

    def mark_real(self) -> None:
        self._mark_candidate(TargetVerdict.REAL, "真目标")

    def mark_bogus(self) -> None:
        self._mark_candidate(TargetVerdict.BOGUS, "假目标")

    def next_candidate(self) -> None:
        if not self._window._candidates:
            return

        self._window._current_candidate_idx = (
            (self._window._current_candidate_idx + 1) % len(self._window._candidates)
        )
        self.focus_candidate(self._window._current_candidate_idx)

    def candidate_selected(self, index: int) -> None:
        self._window._current_candidate_idx = index
        self.focus_candidate(index)

    def candidate_double_clicked(self, index: int) -> None:
        if 0 <= index < len(self._window._candidates):
            candidate = self._window._candidates[index]
            self._window._current_candidate_idx = index
            self._window.image_viewer.center_on_point(
                candidate.x,
                candidate.y,
                zoom_to=200,
            )

    def focus_candidate(self, index: int) -> None:
        if 0 <= index < len(self._window._candidates):
            candidate = self._window._candidates[index]
            self._window.image_viewer.center_on_point(candidate.x, candidate.y)
            self._window._update_markers()
            self._window.status_pixel_coord.set_pixel_coordinates(
                candidate.x,
                candidate.y,
            )

    def _mark_candidate(self, verdict: TargetVerdict, label: str) -> None:
        if not self._window._candidates or self._window._current_candidate_idx < 0:
            return
        if self._window._current_candidate_idx >= len(self._window._candidates):
            return

        candidate = self._window._candidates[self._window._current_candidate_idx]
        candidate.verdict = verdict
        self._window.suspect_table.update_candidate(self._window._current_candidate_idx)
        self._window._update_markers()
        self._window._show_message(
            f"候选 #{self._window._current_candidate_idx + 1} → {label}"
        )
