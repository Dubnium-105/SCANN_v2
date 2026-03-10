"""查询流程控制器。"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from PyQt5.QtWidgets import QApplication, QMenu

from scann.core.astrometry import format_dec_dms, format_ra_hms, pixel_to_wcs
from scann.core.models import Candidate, TargetVerdict
from scann.core.observation_report import Observation, generate_mpc_report
from scann.gui.dialogs.query_result_popup import QueryResultPopup
from scann.services.query_service import QueryResult, QueryService

if TYPE_CHECKING:
    from scann.gui.main_window import MainWindow


class QueryController:
    """集中主窗口中的查询、复制坐标和报告入口。"""

    def __init__(
        self,
        window: MainWindow,
        query_service: QueryService | None = None,
    ) -> None:
        self._window = window
        self._query_service = query_service

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
        if self._window._new_fits_header is not None:
            sky = pixel_to_wcs(x, y, self._window._new_fits_header)
            if sky:
                ra_deg = sky.ra
                dec_deg = sky.dec
                self._window._show_message(
                    f"正在查询 {query_type} (RA={ra_deg:.4f}, Dec={dec_deg:.4f})...",
                    5000,
                )

                service = self._query_service or QueryService()
                results = self._execute_query(
                    service,
                    query_type,
                    ra_deg,
                    dec_deg,
                )
                self._show_query_results(query_type, ra_deg, dec_deg, results)
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

                sky = pixel_to_wcs(int(candidate.x), int(candidate.y), header)
                if sky is None:
                    continue

                observations.append(
                    Observation(
                        designation="",
                        discovery=False,
                        obs_datetime=obs_dt,
                        ra_deg=sky.ra,
                        dec_deg=sky.dec,
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
        self._window.candidate_presenter.set_candidates(self._window._candidates)
        self._window._update_markers()
        self._window._show_message(f"已添加手动候选体 ({x}, {y})")

    def copy_wcs_coordinates(self, x: int, y: int) -> None:
        if self._window._new_fits_header is None:
            self._window._show_message("无 WCS 头信息，无法转换坐标")
            return

        sky = pixel_to_wcs(x, y, self._window._new_fits_header)
        if sky:
            text = f"{format_ra_hms(sky.ra)}  {format_dec_dms(sky.dec)}"
            QApplication.clipboard().setText(text)
            self._window._show_message(f"已复制: {text}")
            return

        self._window._show_message("WCS 转换失败")

    def _execute_query(
        self,
        service: QueryService,
        query_type: str,
        ra_deg: float,
        dec_deg: float,
    ) -> list[QueryResult]:
        try:
            if query_type == "vsx":
                return service.query_vsx(ra_deg, dec_deg)
            if query_type == "mpc":
                return service.query_mpc(ra_deg, dec_deg)
            if query_type == "simbad":
                return service.query_simbad(ra_deg, dec_deg)
            if query_type == "tns":
                return service.query_tns(ra_deg, dec_deg)
            if query_type == "satellite":
                obs_datetime = None
                if self._window._new_fits_header is not None:
                    obs_datetime = self._window._new_fits_header.observation_datetime
                return service.check_satellite(ra_deg, dec_deg, obs_datetime=obs_datetime)
        except Exception as exc:
            self._window._show_message(f"查询失败: {exc}", 5000, level="WARNING")
            return []

        self._window._show_message(f"不支持的查询类型: {query_type}", 5000, level="WARNING")
        return []

    def _show_query_results(
        self,
        query_type: str,
        ra_deg: float,
        dec_deg: float,
        results: list[QueryResult],
    ) -> None:
        popup = QueryResultPopup(
            title=f"{query_type.upper()} 查询结果",
            parent=self._window,
        )
        if results:
            lines = [
                f"{result.name}  类型={result.object_type}  距离={result.distance_arcsec:.1f}″"
                for result in results
            ]
            popup.set_content(
                "\n".join(lines),
                coords=f"RA={ra_deg:.4f}  Dec={dec_deg:.4f}",
            )
            popup.set_success(count=len(results))
        else:
            popup.set_content(
                "未找到匹配天体",
                coords=f"RA={ra_deg:.4f}  Dec={dec_deg:.4f}",
            )
        popup.show()