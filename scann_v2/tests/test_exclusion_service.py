"""Exclusion Service 测试

使用测试驱动开发 (TDD) 实现：
1. 坐标匹配逻辑：将候选体与已知小行星匹配
2. Siril 天体测量 + PSF 坐标解析
3. 候选体坐标缓存
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from scann.core.models import Candidate, CandidateFeatures, FitsHeader, ObservatoryConfig, SkyPosition
from scann.core.mpcorb import AsteroidOrbit
from scann.services.exclusion_service import ExclusionService
from scann.services.siril_astrometry import CandidateSkyCoordinateCache, ResolvedSkyCoordinate


class TestExclusionServiceMatching:
    """测试坐标匹配功能"""

    def test_candidate_marked_as_known_when_matches_asteroid(self):
        service = ExclusionService(
            observatory=ObservatoryConfig(),
            limit_magnitude=20.0,
        )

        mock_asteroid = Mock(spec=AsteroidOrbit)
        mock_asteroid.designation = "2024 ABC"
        mock_asteroid.ra = 180.0
        mock_asteroid.dec = 0.0
        mock_asteroid.mag = 18.5
        service._asteroids = [mock_asteroid]

        candidates = [Candidate(x=100, y=100, features=CandidateFeatures())]
        header = FitsHeader(raw={"RA": 180.0, "DEC": 0.0})

        with patch.object(service, "_pixel_to_sky", return_value=SkyPosition(ra=180.0, dec=0.0)):
            result = service.check_candidates(candidates, header)

        assert len(result) == 1
        assert result[0].is_known is True
        assert result[0].known_id == "2024 ABC"

    def test_candidate_not_marked_when_no_match(self):
        service = ExclusionService()

        mock_asteroid = Mock(spec=AsteroidOrbit)
        mock_asteroid.designation = "2024 ABC"
        mock_asteroid.ra = 180.0
        mock_asteroid.dec = 0.0
        mock_asteroid.mag = 18.5
        service._asteroids = [mock_asteroid]

        candidates = [Candidate(x=200, y=200, features=CandidateFeatures())]
        header = FitsHeader(raw={})

        with patch.object(service, "_pixel_to_sky", return_value=SkyPosition(ra=181.0, dec=1.0)):
            result = service.check_candidates(candidates, header)

        assert len(result) == 1
        assert result[0].is_known is False
        assert result[0].known_id == ""

    def test_pixel_to_sky_uses_cached_siril_coordinate_when_image_path_provided(self):
        resolver = Mock()
        resolver.resolve.return_value = ResolvedSkyCoordinate.from_hms_dms(
            ra_hms="10h49m28.34s",
            dec_dms="+34°43'01.27\"",
        )
        service = ExclusionService(
            coordinate_resolver=resolver,
            coordinate_cache=CandidateSkyCoordinateCache(),
        )
        header = FitsHeader(raw={"RA": 180.0, "DEC": 0.0})
        image_path = Path("/tmp/example.fit")

        first = service.get_candidate_sky_coordinate(header, 1315.45, 615.58, image_path=image_path)
        second = service.get_candidate_sky_coordinate(header, 1315.45, 615.58, image_path=image_path)

        assert first.position.ra == pytest.approx(second.position.ra)
        assert first.normalized_coordinate == "10 49 28.34 +34 43 01.27"
        resolver.resolve.assert_called_once_with(image_path, 1315.45, 615.58)

    def test_pixel_to_sky_falls_back_to_header_when_siril_resolution_fails(self):
        resolver = Mock()
        resolver.resolve.side_effect = RuntimeError("siril failed")
        service = ExclusionService(coordinate_resolver=resolver)
        header = FitsHeader(
            raw={
                "CRVAL1": 180.0,
                "CRVAL2": 0.0,
                "CRPIX1": 100.0,
                "CRPIX2": 100.0,
                "CDELT1": 1.0 / 3600.0,
                "CDELT2": 1.0 / 3600.0,
            }
        )

        position = service._pixel_to_sky(header, 101, 101, image_path=Path("/tmp/example.fit"))

        assert position.ra == pytest.approx(180.0 + 1.0 / 3600.0, abs=1e-6)
        assert position.dec == pytest.approx(1.0 / 3600.0, abs=1e-6)

    def test_pixel_to_sky_raises_when_header_has_no_usable_coordinate_fallback(self):
        resolver = Mock()
        resolver.resolve.side_effect = RuntimeError("siril failed")
        service = ExclusionService(coordinate_resolver=resolver)
        header = FitsHeader(raw={})

        with pytest.raises(ValueError, match="天球坐标"):
            service.get_candidate_sky_coordinate(
                header,
                101,
                101,
                image_path=Path("/tmp/example.fit"),
            )

    def test_matching_within_radius(self):
        service = ExclusionService()

        mock_asteroid = Mock(spec=AsteroidOrbit)
        mock_asteroid.designation = "2024 XYZ"
        mock_asteroid.ra = 180.0
        mock_asteroid.dec = 0.0
        mock_asteroid.mag = 18.0
        service._asteroids = [mock_asteroid]

        candidates = [Candidate(x=100, y=100, features=CandidateFeatures())]
        header = FitsHeader(raw={})

        with patch.object(service, "_pixel_to_sky", return_value=SkyPosition(ra=180.0 + 3.0 / 3600.0, dec=0.0)):
            result = service.check_candidates(candidates, header)

        assert result[0].is_known is True

    def test_no_matching_outside_radius(self):
        service = ExclusionService()

        mock_asteroid = Mock(spec=AsteroidOrbit)
        mock_asteroid.designation = "2024 XYZ"
        mock_asteroid.ra = 180.0
        mock_asteroid.dec = 0.0
        mock_asteroid.mag = 18.0
        service._asteroids = [mock_asteroid]

        candidates = [Candidate(x=100, y=100, features=CandidateFeatures())]
        header = FitsHeader(raw={})

        with patch.object(service, "_pixel_to_sky", return_value=SkyPosition(ra=180.0 + 10.0 / 3600.0, dec=0.0)):
            result = service.check_candidates(candidates, header)

        assert result[0].is_known is False