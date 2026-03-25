"""Tests for known-object exclusion."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from scann.core.models import Candidate, CandidateFeatures, FitsHeader, ObservatoryConfig, SkyPosition
from scann.core.mpcorb import AsteroidOrbit
from scann.services.exclusion_service import ExclusionService
from scann.services.siril_astrometry import CandidateSkyCoordinateCache, ResolvedSkyCoordinate


class TestExclusionServiceMatching:
    def test_candidate_marked_as_known_when_matches_static_coordinate(self) -> None:
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

    @patch("scann.services.exclusion_service.compute_apparent_positions")
    def test_candidate_marked_as_known_when_matches_propagated_position(self, mock_compute) -> None:
        service = ExclusionService(observatory=ObservatoryConfig(code="X99"))
        asteroid = AsteroidOrbit(
            designation="2024 ABC",
            epoch=2459000.5,
            mean_anomaly=45.0,
            arg_perihelion=0.0,
            ascending_node=0.0,
            inclination=0.0,
            eccentricity=0.0,
            semi_major_axis=2.0,
            abs_magnitude=18.5,
            slope_param=0.15,
        )
        service._asteroids = [asteroid]
        mock_compute.return_value = [
            SkyPosition(ra=180.0, dec=0.0, mag=18.5, name="2024 ABC"),
        ]

        candidates = [Candidate(x=100, y=100, features=CandidateFeatures())]
        header = FitsHeader(
            raw={
                "DATE-OBS": "2020-05-31T12:00:00",
                "EXPTIME": 120.0,
            }
        )

        with patch.object(service, "_pixel_to_sky", return_value=SkyPosition(ra=180.0, dec=0.0)):
            result = service.check_candidates(candidates, header)

        assert result[0].is_known is True
        assert result[0].known_id == "2024 ABC"
        mock_compute.assert_called_once()
        args = mock_compute.call_args.args
        assert args[0] == [asteroid]
        assert args[1] == datetime(2020, 5, 31, 12, 1, 0)
        assert args[2] == service.observatory

    @patch("scann.services.exclusion_service.compute_apparent_positions")
    def test_falls_back_to_static_coordinates_when_propagation_fails(self, mock_compute) -> None:
        service = ExclusionService()
        mock_compute.side_effect = RuntimeError("propagation failed")

        mock_asteroid = Mock(spec=AsteroidOrbit)
        mock_asteroid.designation = "2024 ABC"
        mock_asteroid.ra = 180.0
        mock_asteroid.dec = 0.0
        mock_asteroid.mag = 18.5
        mock_asteroid.abs_magnitude = 18.5
        service._asteroids = [mock_asteroid]

        candidates = [Candidate(x=100, y=100, features=CandidateFeatures())]
        header = FitsHeader(raw={"DATE-OBS": "2020-05-31T12:00:00"})

        with patch.object(service, "_pixel_to_sky", return_value=SkyPosition(ra=180.0, dec=0.0)):
            result = service.check_candidates(candidates, header)

        assert result[0].is_known is True
        assert result[0].known_id == "2024 ABC"

    @patch("scann.services.exclusion_service.compute_apparent_positions")
    def test_candidate_not_marked_when_no_match(self, mock_compute) -> None:
        service = ExclusionService()
        asteroid = AsteroidOrbit(
            designation="2024 ABC",
            epoch=2459000.5,
            mean_anomaly=0.0,
            arg_perihelion=0.0,
            ascending_node=0.0,
            inclination=0.0,
            eccentricity=0.0,
            semi_major_axis=2.0,
            abs_magnitude=18.5,
            slope_param=0.15,
        )
        service._asteroids = [asteroid]
        mock_compute.return_value = [
            SkyPosition(ra=180.0, dec=0.0, mag=18.5, name="2024 ABC"),
        ]

        candidates = [Candidate(x=200, y=200, features=CandidateFeatures())]
        header = FitsHeader(raw={"DATE-OBS": "2020-05-31T12:00:00"})

        with patch.object(service, "_pixel_to_sky", return_value=SkyPosition(ra=181.0, dec=1.0)):
            result = service.check_candidates(candidates, header)

        assert len(result) == 1
        assert result[0].is_known is False
        assert result[0].known_id == ""

    def test_pixel_to_sky_uses_cached_siril_coordinate_when_image_path_provided(self) -> None:
        resolver = Mock()
        resolver.resolve.return_value = ResolvedSkyCoordinate.from_decimal_degrees(
            ra=162.36808333333334,
            dec=34.71701944444444,
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

    def test_pixel_to_sky_falls_back_to_header_when_siril_resolution_fails(self) -> None:
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

    def test_pixel_to_sky_raises_when_header_has_no_usable_coordinate_fallback(self) -> None:
        resolver = Mock()
        resolver.resolve.side_effect = RuntimeError("siril failed")
        service = ExclusionService(coordinate_resolver=resolver)
        header = FitsHeader(raw={})

        with pytest.raises(ValueError, match="sky coordinate information"):
            service.get_candidate_sky_coordinate(
                header,
                101,
                101,
                image_path=Path("/tmp/example.fit"),
            )

    def test_matching_within_radius(self) -> None:
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

    def test_no_matching_outside_radius(self) -> None:
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
