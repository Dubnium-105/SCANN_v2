"""配对流程控制器。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt5.QtWidgets import QFileDialog

from scann.core.dataset_storage import DatasetStorage
from scann.data.file_manager import FitsImagePair
from scann.services.dataset_preprocess_service import DatasetPreprocessService
from scann.services.pair_service import PairService

if TYPE_CHECKING:
    from scann.gui.main_window import MainWindow


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
        self._reset_loaded_dataset_state()

        if created:
            created_desc = ", ".join(created)
            self._window._show_message(
                f"已自动创建数据集目录结构: {dataset_root} ({created_desc})",
                5000,
            )

        self._window._show_message(f"正在预处理数据集: {dataset_root}", 2000)
        report = self._preprocess_service.prepare_dataset(dataset_root)
        pair_count, only_new_count, only_old_count = self._refresh_dataset_listing(dataset_root)

        self._window._show_message(
            f"{message_prefix}: "
            f"{dataset_root} · 总任务 {report.total_task_count} · "
            f"就绪 {report.task_count} · 对齐失败 {report.align_failed_count} · "
            f"配对 {pair_count} · 仅新图 {only_new_count} · 仅旧图 {only_old_count}",
            5000,
        )

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
            marked_path = self._pair_service.resolve_marked_image_path(pair.new_path)
            if marked_path is not None:
                marked_image = self._pair_service.read_image(marked_path)
                self._window._new_marked_image_data = marked_image.data
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
