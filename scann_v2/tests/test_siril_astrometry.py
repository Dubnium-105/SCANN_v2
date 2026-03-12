from pathlib import Path
import shutil
from unittest.mock import Mock

import pytest

from scann.services.siril_astrometry import (
    CandidateSkyCoordinateCache,
    ResolvedSkyCoordinate,
    SirilAstrometryResolver,
    parse_findstar_catalog,
    parse_psf_sky_coordinate,
)


def test_parse_psf_sky_coordinate_preserves_two_decimals() -> None:
    output = """
    质心坐标：
            x0=1315.45px   10h49m28.34s J2000
            y0=615.58px   +34°43'01.27\" J2000
    """

    resolved = parse_psf_sky_coordinate(output)

    assert resolved.raw_coordinate == '10h49m28.34s+34°43\'01.27"'
    assert resolved.normalized_coordinate == "10 49 28.34 +34 43 01.27"
    assert resolved.position.ra == pytest.approx(162.3680833, abs=1e-6)
    assert resolved.position.dec == pytest.approx(34.7170194, abs=1e-6)


def test_parse_findstar_catalog_selects_nearest_detection_with_two_decimal_display() -> None:
    catalog = """# star#\tlayer\tB\tA\tbeta\tX\tY\tFWHMx [px]\tFWHMy [px]\tFWHMx [\"]\tFWHMy [\"]\tangle\tRMSE\tmag\tSat\tProfile\tRA\tDec
20\t0\t0\t0\t-1\t1300.00\t600.00\t2.0\t2.0\t4.0\t4.0\t0\t0\t0\t0\tGaussian\t162.300000\t34.700000
21\t0\t0\t0\t-1\t1312.34\t615.63\t2.0\t2.0\t4.0\t4.0\t0\t0\t0\t0\tGaussian\t162.366960\t34.716899
"""

    resolved = parse_findstar_catalog(catalog, target_x=1315.45, target_y=615.58, selection_size=20)

    assert resolved.raw_coordinate == '10h49m28.07s+34°43\'00.84"'
    assert resolved.normalized_coordinate == "10 49 28.07 +34 43 00.84"
    assert resolved.position.ra == pytest.approx(162.36696, abs=1e-9)
    assert resolved.position.dec == pytest.approx(34.716899, abs=1e-9)


def test_coordinate_cache_reuses_resolved_result() -> None:
    cache = CandidateSkyCoordinateCache()
    resolver = Mock()
    resolver.resolve.return_value = ResolvedSkyCoordinate.from_hms_dms(
        ra_hms="10h49m28.34s",
        dec_dms="+34°43'01.27\"",
    )
    image_path = Path("/tmp/example.fit")

    first = cache.get_or_resolve(image_path, 1315, 616, resolver)
    second = cache.get_or_resolve(image_path, 1315, 616, resolver)

    assert first == second
    resolver.resolve.assert_called_once_with(image_path, 1315, 616)


def test_resolver_builds_script_with_fixed_astrometry_settings() -> None:
    runner = Mock(return_value="""
    x0=1315.45px   10h49m28.00s J2000
    y0=615.58px   +34°43'01.00\" J2000
    """)
    resolver = SirilAstrometryResolver(run_script_fn=runner)
    image_path = Path("C:/data/example.fit")

    resolver.resolve(image_path, 1315.45, 615.58)

    working_dir, script = runner.call_args.args
    assert working_dir == image_path.parent
    assert 'load "example.fit"' in script
    assert "platesolve -force -focal=2042.6 -pixelsize=18.0" in script
    assert "setfindstar reset -focal=2042.6 -pixelsize=18.0" in script
    assert "findstar -out=scann_findstar_" in script
    assert "boxselect 1305 606 20 20" in script
    assert "psf" in script


@pytest.mark.integration
def test_resolver_real_sample_preserves_fractional_seconds() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sample_path = repo_root / "dataset" / "new" / "NGC 3381__aligned_crop.fts"

    if not sample_path.exists():
        pytest.skip("真实样本不存在")
    if shutil.which("siril-cli.exe") is None and shutil.which("siril.exe") is None:
        pytest.skip("Siril CLI 不可用")

    resolver = SirilAstrometryResolver()

    resolved = resolver.resolve(sample_path, 1315.45, 615.58)

    assert resolved.normalized_coordinate == "10 49 28.07 +34 43 00.84"
    assert resolved.position.ra == pytest.approx(162.366960, abs=1e-6)
    assert resolved.position.dec == pytest.approx(34.716899, abs=1e-6)
