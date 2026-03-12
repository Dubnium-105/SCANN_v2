"""查询流程控制器。"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMenu

from scann.core.astrometry import pixel_to_wcs
from scann.core.models import Candidate, TargetVerdict
from scann.core.observation_report import Observation, generate_mpc_report
from scann.gui.dialogs.query_result_popup import QueryResultPopup
from scann.services.query_service import QueryResponse, QueryService
from scann.services.siril_astrometry import ResolvedSkyCoordinate

if TYPE_CHECKING:
    from scann.gui.main_window import MainWindow


class QueryWorker(QThread):
    finished_with_response = pyqtSignal(str, str, object)

    def __init__(
        self,
        service: QueryService,
        query_type: str,
        coordinate_text: str,
        ra_deg: float,
        dec_deg: float,
        obs_datetime,
        observatory,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._query_type = query_type
        self._coordinate_text = coordinate_text
        self._ra_deg = ra_deg
        self._dec_deg = dec_deg
        self._obs_datetime = obs_datetime
        self._observatory = observatory

    def run(self) -> None:
        response = self._service.execute_query(
            self._query_type,
            self._ra_deg,
            self._dec_deg,
            obs_datetime=self._obs_datetime,
            observatory=self._observatory,
        )
        self.finished_with_response.emit(
            self._query_type,
            self._coordinate_text,
            response,
        )


class QueryController:
    """集中主窗口中的查询、复制坐标和报告入口。"""

    def __init__(
        self,
        window: MainWindow,
        query_service: QueryService | None = None,
    ) -> None:
        self._window = window
        self._query_service = query_service
        self._query_worker: QueryWorker | None = None

    def populate_candidate_coordinates(self, candidates: list[Candidate]) -> None:
        for candidate in candidates:
            resolved = self._resolve_candidate_coordinate(candidate.x, candidate.y)
            text = resolved.text if resolved is not None else "--"
            setattr(candidate, "wcs_text", text)
            setattr(candidate, "sky_position", (resolved.sky if resolved is not None else None))

    def image_clicked(self, x: int, y: int) -> None:
        self._window.status_pixel_coord.set_pixel_coordinates(x, y)

    def image_right_click(self, x: int, y: int) -> None:
        menu = QMenu(self._window)

        queries = [
            ("🔍 查询 VSX", "vsx"),
            ("🔍 查询 MPC", "mpc"),
            ("🔍 查询 SIMBAD", "simbad"),
            ("🔍 查询 TNS", "tns"),
            ("🛰️ 查询人造卫星", "satellite"),
        ]
        for label, query_type in queries:
            action = menu.addAction(label)
            action.triggered.connect(
                lambda checked, current_query=query_type: self.do_query(current_query, x, y)
            )

        menu.addSeparator()
        act_mpc = menu.addAction("📝 生成 MPC 80列报告")
        act_mpc.triggered.connect(
            lambda checked, current_x=x, current_y=y: self.context_mpc_report(
                current_x,
                current_y,
            )
        )
        menu.addSeparator()
        act_add_candidate = menu.addAction("➕ 手动添加候选体")
        act_add_candidate.triggered.connect(
            lambda checked, current_x=x, current_y=y: self.context_add_candidate(
                current_x,
                current_y,
            )
        )
        menu.addSeparator()

        act_copy_pixel = menu.addAction("📋 复制像素坐标")
        act_copy_pixel.triggered.connect(
            lambda: QApplication.clipboard().setText(f"{x}, {y}")
        )
        act_copy_wcs = menu.addAction("📋 复制天球坐标")
        act_copy_wcs.triggered.connect(
            lambda checked, current_x=x, current_y=y: self.copy_wcs_coordinates(
                current_x,
                current_y,
            )
        )

        menu.exec_(
            self._window.image_viewer.mapToGlobal(
                self._window.image_viewer.mapFromScene(float(x), float(y))
            )
        )

    def do_query(self, query_type: str, x: int, y: int) -> None:
        resolved = self._resolve_candidate_coordinate(x, y)
        if resolved is not None:
                ra_deg = resolved.sky.ra
                dec_deg = resolved.sky.dec
                self._window._show_message(
                    f"正在查询 {query_type} ({resolved.text})...",
                    5000,
                )

                service = self._query_service or QueryService()
                if query_type == "satellite":
                    self._start_async_query(
                        service,
                        query_type,
                        ra_deg,
                        dec_deg,
                        resolved.text,
                    )
                    return

                response = self._execute_query(
                    service,
                    query_type,
                    ra_deg,
                    dec_deg,
                )
                self._show_query_results(query_type, resolved.text, response)
                return

        self._window._show_message(
            f"正在查询 {query_type} ({x}, {y})... (无WCS信息，使用像素坐标)",
            5000,
        )

    def menu_query(self, query_type: str) -> None:
        if self._window._candidates and 0 <= self._window._current_candidate_idx < len(
            self._window._candidates
        ):
            candidate = self._window._candidates[self._window._current_candidate_idx]
            self.do_query(query_type, int(candidate.x), int(candidate.y))
            return

        self._window._show_message("请先选中一个候选体，或在图像上右键进行坐标查询")

    def mpc_report(self) -> None:
        from scann.gui.dialogs.mpc_report_dialog import MpcReportDialog

        dialog = MpcReportDialog(self._window)

        if self._window._candidates and self._window._new_fits_header is not None:
            observations = []
            header = self._window._new_fits_header
            obs_dt = header.observation_datetime or datetime.utcnow()
            obs_code = header.raw.get("OBSERVAT", "")[:3] if header.raw.get("OBSERVAT") else ""

            for candidate in self._window._candidates:
                if candidate.verdict == TargetVerdict.BOGUS:
                    continue

                resolved = self._resolve_candidate_coordinate(int(candidate.x), int(candidate.y))
                if resolved is None:
                    continue

                observations.append(
                    Observation(
                        designation="",
                        discovery=False,
                        obs_datetime=obs_dt,
                        ra_deg=resolved.sky.ra,
                        dec_deg=resolved.sky.dec,
                        magnitude=0.0,
                        mag_band="C",
                        observatory_code=obs_code,
                    )
                )

            if observations:
                dialog.set_report(generate_mpc_report(observations))
        elif self._window._new_fits_header is None and self._window._candidates:
            self._window._show_message("无 WCS 头信息，无法生成 MPC 报告坐标")

        dialog.exec_()

    def context_mpc_report(self, x: int, y: int) -> None:
        best_idx = -1
        best_dist = float("inf")
        for index, candidate in enumerate(self._window._candidates):
            distance = (candidate.x - x) ** 2 + (candidate.y - y) ** 2
            if distance < best_dist:
                best_dist = distance
                best_idx = index

        if best_idx >= 0 and best_dist < 50 ** 2:
            self._window._current_candidate_idx = best_idx
            self._window.detection_controller.focus_candidate(best_idx)

        self.mpc_report()

    def context_add_candidate(self, x: int, y: int) -> None:
        candidate = Candidate(
            x=x,
            y=y,
            is_manual=True,
            verdict=TargetVerdict.UNKNOWN,
        )
        self._window._candidates.append(candidate)
        self._window._current_candidate_idx = len(self._window._candidates) - 1
        self.populate_candidate_coordinates(self._window._candidates)
        self._window.candidate_presenter.set_candidates(self._window._candidates)
        self._window._update_markers()
        self._window._show_message(f"已添加手动候选体 ({x}, {y})")

    def copy_wcs_coordinates(self, x: int, y: int) -> None:
        resolved = self._resolve_candidate_coordinate(x, y)
        if resolved is not None:
            text = resolved.text
            QApplication.clipboard().setText(text)
            self._window._show_message(f"已复制: {text}")
            return

        self._window._show_message("WCS 转换失败")

    def _resolve_candidate_coordinate(self, x: int, y: int):
        header = getattr(self._window, "_new_fits_header", None)
        if header is None:
            return None

        detection_controller = getattr(self._window, "detection_controller", None)
        if detection_controller is not None:
            try:
                exclusion_service = detection_controller.get_exclusion_service()
                image_path = detection_controller.resolve_current_new_image_path()
                resolved = exclusion_service.get_candidate_sky_coordinate(
                    header,
                    x,
                    y,
                    image_path=image_path,
                )
                sky = getattr(resolved, "position", None)
                text = getattr(resolved, "normalized_coordinate", None)
                if sky is not None and isinstance(text, str):
                    ra = getattr(sky, "ra", None)
                    dec = getattr(sky, "dec", None)
                    if isinstance(ra, (int, float)) and isinstance(dec, (int, float)):
                        return SimpleNamespace(
                            sky=sky,
                            text=text,
                        )
            except Exception:
                pass

        sky = pixel_to_wcs(x, y, header)
        if sky is None:
            return None
        resolved = ResolvedSkyCoordinate.from_decimal_degrees(sky.ra, sky.dec)
        return SimpleNamespace(
            sky=sky,
            text=resolved.normalized_coordinate,
        )

    def _execute_query(
        self,
        service: QueryService,
        query_type: str,
        ra_deg: float,
        dec_deg: float,
    ) -> QueryResponse:
        obs_datetime = None
        if self._window._new_fits_header is not None:
            obs_datetime = self._window._new_fits_header.observation_datetime
        observatory = self._resolve_observatory()
        kwargs = {"obs_datetime": obs_datetime}
        if observatory is not None:
            kwargs["observatory"] = observatory
        return service.execute_query(
            query_type,
            ra_deg,
            dec_deg,
            **kwargs,
        )

    def _start_async_query(
        self,
        service: QueryService,
        query_type: str,
        ra_deg: float,
        dec_deg: float,
        coordinate_text: str,
    ) -> None:
        if self._query_worker is not None and self._query_worker.isRunning():
            self._window._show_message("已有查询正在进行，请稍候", 5000)
            return

        obs_datetime = None
        if self._window._new_fits_header is not None:
            obs_datetime = self._window._new_fits_header.observation_datetime

        observatory = self._resolve_observatory()

        worker = QueryWorker(
            service,
            query_type,
            coordinate_text,
            ra_deg,
            dec_deg,
            obs_datetime,
            observatory,
            parent=self._window,
        )
        worker.finished_with_response.connect(self._handle_async_query_finished)
        worker.finished.connect(self._clear_query_worker)
        self._query_worker = worker
        worker.start()

    def _handle_async_query_finished(
        self,
        query_type: str,
        coordinate_text: str,
        response: QueryResponse,
    ) -> None:
        self._show_query_results(query_type, coordinate_text, response)

    def _clear_query_worker(self) -> None:
        self._query_worker = None

    def _resolve_observatory(self):
        config = getattr(self._window, "_config", None)
        observatory = getattr(config, "observatory", None) if config is not None else None
        if observatory is None:
            return None

        if any(
            abs(value) > 0.0
            for value in (
                getattr(observatory, "longitude", 0.0),
                getattr(observatory, "latitude", 0.0),
                getattr(observatory, "altitude", 0.0),
            )
        ):
            return observatory

        if getattr(observatory, "code", "") or getattr(observatory, "name", ""):
            return observatory
        return None

    def _show_query_results(
        self,
        query_type: str,
        coordinate_text: str,
        response: QueryResponse,
    ) -> None:
        popup = QueryResultPopup(
            title=f"{query_type.upper()} 查询结果",
            parent=self._window,
        )
        if response.has_error:
            popup.set_error(response.error)
            popup.lbl_coords.setText(coordinate_text)
            self._window._show_message(f"查询失败: {response.error}", 5000, level="WARNING")
        elif response:
            lines = [
                f"{result.name}  类型={result.object_type}  距离={result.distance_arcsec:.1f}″"
                for result in response
            ]
            popup.set_content(
                "\n".join(lines),
                coords=coordinate_text,
            )
            popup.set_success(count=len(response))
        else:
            popup.set_content(
                "未找到匹配天体",
                coords=coordinate_text,
            )
        popup.show()