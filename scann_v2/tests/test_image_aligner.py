from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from scann.core.image_aligner import _align_siril, align
from scann.core.models import AlignResult
from scann.services.pair_service import PairService


class TestImageAlignCrop:
    def test_calc_overlap_crop_bounds_prefers_direct_aligned_overlap(self) -> None:
        service = PairService()
        w, h = 12, 10
        new_image = np.ones((h, w), dtype=np.float32)
        aligned_old = np.ones((h, w), dtype=np.float32)
        aligned_old[:, :4] = 0.0
        aligned_old[:2, :] = 0.0

        crop_bounds = service.calc_overlap_crop_bounds(
            w=w,
            h=h,
            dx=0.0,
            dy=0.0,
            aligned_old=aligned_old,
            new_image=new_image,
        )

        assert crop_bounds is not None
        x0, x1, y0, y1 = crop_bounds
        assert x0 >= 4
        assert y0 >= 2
        assert x1 <= w
        assert y1 <= h

    def test_calc_overlap_crop_bounds_removes_edge_floor_band(self) -> None:
        service = PairService()
        h, w = 12, 14
        new_image = np.full((h, w), 1200.0, dtype=np.float32)
        aligned_old = np.full((h, w), 1200.0, dtype=np.float32)
        aligned_old[:, :3] = 862.0
        aligned_old[:2, :] = 862.0

        crop_bounds = service.calc_overlap_crop_bounds(
            w=w,
            h=h,
            dx=0.0,
            dy=0.0,
            aligned_old=aligned_old,
            new_image=new_image,
        )

        assert crop_bounds is not None
        x0, x1, y0, y1 = crop_bounds
        assert x0 >= 3
        assert y0 >= 2
        assert x1 <= w
        assert y1 <= h

    def test_calc_overlap_crop_bounds_falls_back_to_geometry(self) -> None:
        service = PairService()

        crop_bounds = service.calc_overlap_crop_bounds(
            w=10,
            h=10,
            dx=3.0,
            dy=0.0,
            aligned_old=np.zeros((10, 10), dtype=np.float32),
        )

        assert crop_bounds == (3, 10, 0, 10)

    def test_calc_overlap_crop_bounds_no_overlap(self) -> None:
        service = PairService()

        crop_bounds = service.calc_overlap_crop_bounds(
            w=10,
            h=10,
            dx=15.0,
            dy=0.0,
            aligned_old=np.ones((10, 10), dtype=np.float32),
        )

        assert crop_bounds is None


class TestSirilOnlyAlignment:
    def test_rejects_removed_local_methods(self) -> None:
        image = np.ones((8, 8), dtype=np.float32)

        result = align(image, image, method="phase_correlation")

        assert not result.success
        assert result.aligned_old is None
        assert "Only Siril" in result.error_message

    def test_auto_is_siril_compatibility_alias(self, monkeypatch) -> None:
        image = np.arange(16, dtype=np.float32).reshape(4, 4)
        aligned = image + 10
        calls: list[str] = []

        def fake_siril(reference, moving, method="siril", max_shift=100):
            calls.append(method)
            return AlignResult(aligned_old=aligned, dx=0.0, dy=0.0, success=True)

        monkeypatch.setattr("scann.core.image_aligner._align_siril", fake_siril)

        result = align(image, image.copy(), method="auto", max_shift=32)

        assert result.success
        assert result.rotation == pytest.approx(0.0)
        assert calls == ["siril"]
        np.testing.assert_array_equal(result.aligned_old, aligned)

    def test_align_prefers_detected_rot180_orientation(self, monkeypatch) -> None:
        reference = np.arange(25, dtype=np.float32).reshape(5, 5)
        moving = np.rot90(reference, 2)
        calls: list[np.ndarray] = []

        def fake_siril(reference_image, moving_image, method="siril", max_shift=100):
            calls.append(np.asarray(moving_image, dtype=np.float32))
            if np.array_equal(moving_image, reference_image):
                return AlignResult(aligned_old=moving_image.copy(), success=True)
            return AlignResult(aligned_old=None, success=False, error_message="orientation failed")

        monkeypatch.setattr("scann.core.image_aligner._align_siril", fake_siril)

        result = align(reference, moving, method="siril")

        assert result.success
        assert result.rotation == pytest.approx(180.0)
        assert len(calls) == 1
        np.testing.assert_array_equal(calls[0], reference)

    def test_align_reports_both_siril_orientation_failures(self, monkeypatch) -> None:
        image = np.arange(9, dtype=np.float32).reshape(3, 3)

        def fake_siril(*_args, **_kwargs):
            return AlignResult(aligned_old=None, success=False, error_message="no stars")

        monkeypatch.setattr("scann.core.image_aligner._align_siril", fake_siril)

        result = align(image, image.copy())

        assert not result.success
        assert "original: no stars" in result.error_message
        assert "rot180: no stars" in result.error_message

    def test_siril_script_uses_register_and_relax_off(self, monkeypatch, tmp_path) -> None:
        new_image = np.ones((32, 32), dtype=np.float32)
        old_image = np.ones((32, 32), dtype=np.float32)
        aligned_image = np.ones((32, 32), dtype=np.float32)
        aligned_output = tmp_path / "r_pair_00002.fit"
        captured_scripts: list[str] = []

        monkeypatch.setattr("scann.core.image_aligner._find_siril_executable", lambda: "siril-cli")

        class FakeTempDir:
            def __enter__(self):
                return str(tmp_path)

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakePrimaryHDU:
            def __init__(self, data):
                self.data = data

            def writeto(self, path, overwrite=False):
                Path(path).write_bytes(b"input")

        class FakeOpen:
            def __enter__(self):
                return [type("HDU", (), {"data": aligned_image})()]

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_fits = type(
            "FakeFitsModule",
            (),
            {
                "PrimaryHDU": FakePrimaryHDU,
                "open": staticmethod(lambda *_args, **_kwargs: FakeOpen()),
            },
        )()

        monkeypatch.setitem(sys.modules, "astropy", type("Astropy", (), {})())
        monkeypatch.setitem(sys.modules, "astropy.io", type("AstropyIO", (), {"fits": fake_fits})())
        monkeypatch.setitem(sys.modules, "astropy.io.fits", fake_fits)
        monkeypatch.setattr(
            "scann.core.image_aligner.tempfile.TemporaryDirectory",
            lambda **_kwargs: FakeTempDir(),
        )

        def fake_run(args, **_kwargs):
            aligned_output.write_bytes(b"fake")
            captured_scripts.append(Path(args[4]).read_text(encoding="utf-8"))
            return type("Proc", (), {"returncode": 0, "stdout": b"", "stderr": b""})()

        monkeypatch.setattr("scann.core.image_aligner.subprocess.run", fake_run)

        result = _align_siril(new_image, old_image, max_shift=32)

        assert result.success
        assert result.dx == pytest.approx(0.0)
        assert result.dy == pytest.approx(0.0)
        assert captured_scripts
        assert all("register pair_" in script for script in captured_scripts)
        assert all("-relax=on" not in script for script in captured_scripts)
        assert all("setfindstar reset" in script for script in captured_scripts)
