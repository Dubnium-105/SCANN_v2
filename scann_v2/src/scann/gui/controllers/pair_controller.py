"""配对流程控制器。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QFileDialog

from scann.core.dataset_storage import DatasetStorage
from scann.core.image_aligner import align, align_with_rot180_selection
from scann.data.file_manager import FitsImagePair
from scann.services.dataset_preprocess_service import DatasetPreprocessService
from scann.services.pair_service import PairService

if TYPE_CHECKING:
    from scann.gui.main_window import MainWindow


class DatasetPreprocessWorker(QThread):
    """后台执行数据集预处理，避免阻塞主线程。"""

    progress = pyqtSignal(int, int, str)
    finished_with_report = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        preprocess_service: DatasetPreprocessService,
        dataset_root: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._preprocess_service = preprocess_service
        self._dataset_root = Path(dataset_root)

    def run(self) -> None:
        try:
            if hasattr(self._preprocess_service, "set_progress_callback"):
                self._preprocess_service.set_progress_callback(self._emit_progress)
            report = self._preprocess_service.prepare_dataset(self._dataset_root)
            self.finished_with_report.emit(report)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if hasattr(self._preprocess_service, "set_progress_callback"):
                self._preprocess_service.set_progress_callback(None)

    def _emit_progress(self, current: int, total: int, message: str) -> None:
        self.progress.emit(int(current), int(total), str(message))


class PairController:
    """集中管理主窗口中的数据集加载、预处理、配对切换与图像载入。"""

    def __init__(
        self,
        window: MainWindow,
        pair_service: PairService,
        preprocess_service: DatasetPreprocessService | None = None,
    ) -> None:
        self._window = window
        self._pair_service = pair_service
        self._preprocess_service = preprocess_service or DatasetPreprocessService(
            pair_service=pair_service
        )
        self._last_preprocess_logged_percent = -1
        self._last_preprocess_log_message: str = ""
        self._preprocess_worker: DatasetPreprocessWorker | None = None
        self._preprocess_dataset_root: Path | None = None
        self._preprocess_message_prefix: str = ""

    @property
    def pair_service(self) -> PairService:
        return self._pair_service

    @staticmethod
    def _resolve_dataset_root(selected_path: str | Path) -> Path:
        root = Path(selected_path)
        has_dataset_layout = any((root / name).exists() for name in ("new", "old", "new_marked"))
        if has_dataset_layout or root.name.lower() == "dataset":
            return root
        return root / "dataset"

    @staticmethod
    def _ensure_dataset_dirs(dataset_root: Path) -> list[str]:
        created: list[str] = []
        if not dataset_root.exists():
            dataset_root.mkdir(parents=True, exist_ok=True)
            created.append("dataset")

        for name in (
            "new",
            "old",
            "new_marked",
            "dataset_raw/new",
            "dataset_raw/old",
            "dataset_raw/new_marked",
        ):
            folder = dataset_root / name
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)
                created.append(name)
        return created

    def _reset_loaded_dataset_state(self) -> None:
        self._window.file_list.clear()
        self._window._image_pairs = []
        self._window._current_pair_idx = -1
        self._window._current_pair_using_aligned = False
        self._window._candidates_cache.clear()

    def _activate_dataset_root(self, dataset_root: Path) -> None:
        self._window._dataset_root = str(dataset_root)
        self._window._new_folder = str(dataset_root / "new")
        self._window._old_folder = str(dataset_root / "old")
        self._window._config.new_folder = self._window._new_folder
        self._window._config.old_folder = self._window._old_folder

    def _load_first_new_image(self, files) -> None:
        if not files:
            self._window.set_image_data(None, None, None)
            self._window._new_fits_header = None
            self._window._old_fits_header = None
            return

        first_image = self._pair_service.read_image(files[0].path)
        marked_image = None
        marked_path = self._pair_service.resolve_marked_image_path(files[0].path)
        if marked_path is not None:
            marked_image = self._pair_service.read_image(marked_path)
        self._window.set_image_data(
            first_image.data,
            None,
            marked_image.data if marked_image is not None else None,
        )
        self._window._new_fits_header = first_image.header
        self._window._old_fits_header = None

    def _refresh_dataset_listing(self, dataset_root: Path) -> tuple[int, int, int]:
        prepared_tasks = self._preprocess_service.collect_preprocessed_tasks(dataset_root)
        pairs = [
            FitsImagePair(
                name=task.task_id,
                new_path=task.new_path,
                old_path=task.old_path,
            )
            for task in prepared_tasks
            if task.old_path is not None
        ]
        only_new = [task.task_id for task in prepared_tasks if task.old_path is None]
        only_old: list[str] = []

        self._window._image_pairs = pairs
        self._window._current_pair_idx = -1
        self._window._current_pair_using_aligned = False
        self._window._candidates_cache.clear()

        self._window.file_list.clear()
        for pair in pairs:
            self._window.file_list.addItem(f"✅ {pair.name}")
        for name in only_new:
            self._window.file_list.addItem(f"🆕 {name} (仅新图)")
        for name in only_old:
            self._window.file_list.addItem(f"📲 {name} (仅旧图)")

        if pairs:
            self.load_pair(0)
        else:
            self._window.set_image_data(None, None, None)
            self._window._new_fits_header = None
            self._window._old_fits_header = None

        return len(pairs), len(only_new), len(only_old)

    def _run_preprocess_and_refresh(
        self,
        dataset_root: Path,
        *,
        created: list[str] | None = None,
        message_prefix: str,
    ) -> None:
        if self._preprocess_worker is not None and self._preprocess_worker.isRunning():
            self._window._show_message("已有预处理任务正在运行，请稍候", 3000, level="WARNING")
            return

        self._reset_loaded_dataset_state()
        self._last_preprocess_logged_percent = -1
        self._last_preprocess_log_message = ""
        self._preprocess_dataset_root = dataset_root
        self._preprocess_message_prefix = message_prefix

        if created:
            created_desc = ", ".join(created)
            self._window._show_message(
                f"已自动创建数据集目录结构: {dataset_root} ({created_desc})",
                5000,
            )

        self._set_preprocess_ui_busy(True)
        self._apply_progress(0, 100, f"正在预处理数据集: {dataset_root}", force_log=True)

        if getattr(self._window, "_enable_async_preprocess", False):
            self._start_async_preprocess(dataset_root)
            return

        self._run_sync_preprocess(dataset_root)

    def _format_progress_bar(self, current: int, total: int) -> str:
        total_safe = max(1, total)
        ratio = max(0.0, min(1.0, current / total_safe))
        slots = 20
        filled = int(round(ratio * slots))
        return f"[{'#' * filled}{'-' * (slots - filled)}]"

    def _apply_progress(self, current: int, total: int, message: str, *, force_log: bool = False) -> None:
        safe_total = max(1, int(total))
        safe_current = max(0, min(int(current), safe_total))
        percent = int((safe_current / safe_total) * 100)

        progress_bar = getattr(self._window, "progress_bar", None)
        if progress_bar is not None:
            try:
                progress_bar.setVisible(True)
                progress_bar.setMaximum(safe_total)
                progress_bar.setValue(safe_current)
                progress_bar.setFormat(f"预处理: {percent}%")
            except Exception:
                pass

        should_log = (
            force_log
            or safe_current == safe_total
            or message != self._last_preprocess_log_message
            or percent > self._last_preprocess_logged_percent
        )
        if should_log:
            self._last_preprocess_logged_percent = percent
            self._last_preprocess_log_message = message
            bar_text = self._format_progress_bar(safe_current, safe_total)
            self._window._show_message(f"{message} {bar_text} {percent}%", 0)

        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _set_preprocess_ui_busy(self, busy: bool) -> None:
        progress_bar = getattr(self._window, "progress_bar", None)
        if progress_bar is not None:
            try:
                progress_bar.setVisible(True if busy else False)
            except Exception:
                pass

        for attr_name in (
            "btn_new_folder",
            "btn_preprocess_dataset",
            "btn_align",
            "btn_detect",
            "act_open_new",
            "act_preprocess_dataset",
        ):
            control = getattr(self._window, attr_name, None)
            if control is None:
                continue
            try:
                control.setEnabled(not busy)
            except Exception:
                pass

    def _start_async_preprocess(self, dataset_root: Path) -> None:
        worker = DatasetPreprocessWorker(self._preprocess_service, dataset_root, parent=self._window)
        worker.progress.connect(self._on_preprocess_progress)
        worker.finished_with_report.connect(self._on_preprocess_finished)
        worker.failed.connect(self._on_preprocess_failed)
        worker.finished.connect(self._on_preprocess_worker_finished)
        self._preprocess_worker = worker
        worker.start()

    def _run_sync_preprocess(self, dataset_root: Path) -> None:
        if hasattr(self._preprocess_service, "set_progress_callback"):
            self._preprocess_service.set_progress_callback(self._on_preprocess_progress)
        try:
            report = self._preprocess_service.prepare_dataset(dataset_root)
            self._on_preprocess_finished(report)
        except Exception as exc:
            self._on_preprocess_failed(str(exc))
        finally:
            if hasattr(self._preprocess_service, "set_progress_callback"):
                self._preprocess_service.set_progress_callback(None)

    def _on_preprocess_progress(self, current: int, total: int, message: str) -> None:
        self._apply_progress(current, total, message)

    def _on_preprocess_finished(self, report) -> None:
        dataset_root = self._preprocess_dataset_root
        if dataset_root is None:
            self._on_preprocess_failed("预处理完成但未找到数据集路径")
            return

        self._apply_progress(100, 100, "数据集预处理完成", force_log=True)
        pair_count, only_new_count, only_old_count = self._refresh_dataset_listing(dataset_root)
        self._set_preprocess_ui_busy(False)

        self._window._show_message(
            f"{self._preprocess_message_prefix}: "
            f"{dataset_root} · 总任务 {report.total_task_count} · "
            f"就绪 {report.task_count} · 对齐失败 {report.align_failed_count} · "
            f"配对 {pair_count} · 仅新图 {only_new_count} · 仅旧图 {only_old_count}",
            5000,
        )

    def _on_preprocess_failed(self, message: str) -> None:
        self._set_preprocess_ui_busy(False)
        self._window._show_message(f"数据集预处理失败: {message}", 5000, level="ERROR")

    def _on_preprocess_worker_finished(self) -> None:
        worker = self._preprocess_worker
        self._preprocess_worker = None
        if worker is not None:
            worker.deleteLater()

    def open_dataset(self) -> None:
        folder = QFileDialog.getExistingDirectory(self._window, "选择项目根目录或数据集目录")
        if not folder:
            return
        self.open_dataset_path(folder)

    def open_dataset_path(self, folder: str | Path) -> None:
        selected_root = Path(folder)
        if not selected_root.exists():
            self._window._show_message(f"文件夹不存在: {folder}", 5000, level="WARNING")
            return

        dataset_root = self._resolve_dataset_root(selected_root)
        created = self._ensure_dataset_dirs(dataset_root)
        self._activate_dataset_root(dataset_root)
        self._run_preprocess_and_refresh(
            dataset_root,
            created=created,
            message_prefix="已加载数据集",
        )
        self.add_recent_folder(str(dataset_root))

    def preprocess_current_dataset(self) -> None:
        dataset_root: Path | None = None
        if self._window._dataset_root:
            current_root = Path(self._window._dataset_root)
            if current_root.exists():
                dataset_root = current_root

        if dataset_root is None:
            folder = QFileDialog.getExistingDirectory(self._window, "选择需要预处理的数据集目录")
            if not folder:
                return
            dataset_root = self._resolve_dataset_root(folder)

        created = self._ensure_dataset_dirs(dataset_root)
        self._activate_dataset_root(dataset_root)
        self._run_preprocess_and_refresh(
            dataset_root,
            created=created,
            message_prefix="已完成数据集预处理",
        )
        self.add_recent_folder(str(dataset_root))

    def open_new_folder(self) -> None:
        self.open_dataset()

    def open_old_folder(self) -> None:
        self.open_dataset()

    def add_recent_folder(self, folder: str) -> None:
        recent_folders = self._window._config.recent_folders
        if folder in recent_folders:
            recent_folders.remove(folder)
        recent_folders.insert(0, folder)
        max_count = self._window._config.max_recent_count
        self._window._config.recent_folders = recent_folders[:max_count]
        self.update_recent_menu()

    def update_recent_menu(self) -> None:
        self._window.menu_recent.clear()
        recent = self._window._config.recent_folders
        if not recent:
            self._window.menu_recent.addAction("(无最近打开)")
            return

        for folder in recent:
            action = self._window.menu_recent.addAction(folder)
            action.triggered.connect(
                lambda checked, current_folder=folder: self.open_recent_folder(current_folder)
            )

    def open_recent_folder(self, folder: str) -> None:
        self.open_dataset_path(folder)

    @staticmethod
    def _sanitize_image_data(data) -> np.ndarray:
        return np.nan_to_num(
            np.asarray(data, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _align_marked_to_new(
        self,
        pair: FitsImagePair,
        marked_data,
    ) -> np.ndarray:
        aligned_marked = self._sanitize_image_data(marked_data)
        try:
            raw_new_image = self._pair_service.read_image(pair.new_path)
        except Exception as exc:
            self._window._logger.warning("读取新图原图失败，无法对齐带标记新图: %s (%s)", pair.name, exc)
            return aligned_marked

        raw_new_data = self._sanitize_image_data(raw_new_image.data)
        if raw_new_data.shape[:2] != aligned_marked.shape[:2]:
            self._window._logger.warning(
                "带标记新图与新图原图尺寸不一致，无法执行额外对齐: %s; new=%s marked=%s",
                pair.name,
                raw_new_data.shape,
                aligned_marked.shape,
            )
            return aligned_marked

        max_shift = max(100, int(min(raw_new_data.shape[:2]) * 0.45))
        result, attempt_name, original_score, rotated_score = align_with_rot180_selection(
            raw_new_data,
            aligned_marked,
            method="auto",
            max_shift=max_shift,
            align_fn=align,
        )
        if attempt_name == "rot180" and rotated_score > original_score + 1e-3:
            self._window._logger.info(
                "检测到带标记新图更接近旋转180度版本，优先旋转后对齐: %s (original=%.4f, rot180=%.4f)",
                pair.name,
                original_score,
                rotated_score,
            )
        if result.success and result.aligned_old is not None:
            if attempt_name == "rot180":
                self._window._logger.info("带标记新图旋转180度后对齐成功: %s", pair.name)
            return self._sanitize_image_data(result.aligned_old)

        self._window._logger.warning(
            "带标记新图对齐新图失败，旋转180度兜底后仍未成功: %s (%s)",
            pair.name,
            result.error_message,
        )
        return aligned_marked

    def prev_pair(self) -> None:
        current = self._window.file_list.currentRow()
        if current > 0:
            self._window.file_list.setCurrentRow(current - 1)

    def next_pair(self) -> None:
        current = self._window.file_list.currentRow()
        if current < self._window.file_list.count() - 1:
            self._window.file_list.setCurrentRow(current + 1)

    def select_pair(self, index: int) -> None:
        if index < 0 or index >= len(self._window._image_pairs):
            return
        self.load_pair(index)

    def load_pair(self, index: int) -> None:
        if index < 0 or index >= len(self._window._image_pairs):
            return

        pair = self._window._image_pairs[index]
        dataset_root = Path(self._window._dataset_root) if self._window._dataset_root else None
        if dataset_root is not None:
            DatasetStorage(dataset_root).mark_task_viewed(pair.name)
        self._window._current_pair_idx = index
        self._window._candidates = []
        self._window._current_candidate_idx = -1
        self._window._update_markers()
        self._window.candidate_presenter.set_candidates([])

        try:
            image_pair = self._pair_service.load_pair(pair)
            self._window._new_image_data = image_pair.new_image.data
            self._window._new_marked_image_data = None
            self._window._old_image_data = image_pair.old_image.data
            self._window._new_fits_header = image_pair.new_image.header
            self._window._old_fits_header = image_pair.old_image.header
            self._window._current_pair_using_aligned = bool(image_pair.aligned)
            aligned_marked_path = self._pair_service.resolve_marked_image_path(image_pair.new_image.path)
            raw_marked_path = self._pair_service.resolve_marked_image_path(pair.new_path)
            marked_path = aligned_marked_path or raw_marked_path
            if marked_path is not None:
                marked_image = self._pair_service.read_image(marked_path)
                marked_data = self._sanitize_image_data(marked_image.data)
                if (
                    image_pair.aligned
                    and aligned_marked_path is None
                    and raw_marked_path is not None
                ):
                    marked_data = self._align_marked_to_new(pair, marked_data)
                    crop_bounds = self._pair_service.resolve_alignment_crop_bounds(pair)
                    if crop_bounds is not None:
                        x0, x1, y0, y1 = crop_bounds
                        marked_shape = marked_data.shape[:2]
                        if marked_shape[0] >= y1 and marked_shape[1] >= x1:
                            marked_data = marked_data[y0:y1, x0:x1]
                    else:
                        self._window._logger.warning(
                            "带标记新图缺少对齐裁剪元数据，回退到原始图: %s",
                            pair.name,
                        )
                self._window._new_marked_image_data = marked_data
            if image_pair.aligned and self._window._old_image_data is not None:
                bounds = self._pair_service.calc_nonzero_valid_bounds(self._window._old_image_data)
                if bounds is not None and self._window._new_image_data is not None:
                    x0, x1, y0, y1 = bounds
                    self._window._new_image_data = self._window._new_image_data[y0:y1, x0:x1]
                    self._window._old_image_data = self._window._old_image_data[y0:y1, x0:x1]
                    if self._window._new_marked_image_data is not None:
                        marked_shape = self._window._new_marked_image_data.shape[:2]
                        if marked_shape[0] >= y1 and marked_shape[1] >= x1:
                            self._window._new_marked_image_data = (
                                self._window._new_marked_image_data[y0:y1, x0:x1]
                            )

            self._window.set_image_data(
                self._window._new_image_data,
                self._window._old_image_data,
                self._window._new_marked_image_data,
            )

            if image_pair.aligned:
                self._window._logger.info("加载已对齐裁剪图像: %s", pair.name)

            if index in self._window._candidates_cache:
                self._window.set_candidates(self._window._candidates_cache[index])
        except Exception as exc:
            self._window._show_message(f"加载失败: {exc}", 5000, level="ERROR")

    def aligned_artifact_paths(self, pair) -> tuple[Path, Path, Path, Path]:
        return self._pair_service.aligned_artifact_paths(pair)

    def pair_has_aligned_artifacts(self, pair) -> bool:
        return self._pair_service.pair_has_aligned_artifacts(pair)

    def resolve_pair_image_paths(self, pair) -> tuple[Path, Path, bool]:
        return self._pair_service.resolve_pair_image_paths(pair)

    def calc_nonzero_valid_bounds(
        self,
        image,
    ) -> tuple[int, int, int, int] | None:
        return self._pair_service.calc_nonzero_valid_bounds(image)

    def calc_overlap_crop_bounds(
        self,
        *,
        w: int,
        h: int,
        dx: float,
        dy: float,
        aligned_old=None,
        new_image=None,
    ) -> tuple[int, int, int, int] | None:
        return self._pair_service.calc_overlap_crop_bounds(
            w=w,
            h=h,
            dx=dx,
            dy=dy,
            aligned_old=aligned_old,
            new_image=new_image,
        )

    def resolve_marked_image_path(self, new_image_path: str | Path) -> Path | None:
        return self._pair_service.resolve_marked_image_path(new_image_path)
