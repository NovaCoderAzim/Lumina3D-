"""
Tests for Lumina3D core pipeline features:
  - telemetry ingestion + geodesy
  - metric georeferencing
  - confidence / coverage analytics

These use only numpy (no COLMAP/torch), so they run in any environment.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_synthetic_track_geometry():
    from telemetry.parsers import synthesize_track

    t = synthesize_track(n=60, pattern="linear")
    assert t.is_synthetic is True
    assert len(t.fixes) == 60
    # linear pass designed to span ~300 m
    assert 250 < t.total_length_m < 350
    # ENU anchored at first fix
    assert t.enu[0] == [pytest.approx(0, abs=1e-6)] * 3
    assert t.enu[-1][0] > 250  # moved east


def test_dji_srt_parse(tmp_path):
    from telemetry import TelemetryLoader

    srt = tmp_path / "clip.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:00,033\nGPS(78.0421,27.1751,100.0)\n\n"
        "2\n00:00:00,033 --> 00:00:00,066\nGPS(78.0422,27.1752,101.0)\n"
    )
    tr = TelemetryLoader().load_file(str(srt))
    assert tr.source == "dji_srt"
    assert len(tr.fixes) == 2
    assert tr.fixes[0].latitude == pytest.approx(27.1751)
    assert tr.fixes[0].longitude == pytest.approx(78.0421)


def test_georeference_recovers_known_scale(tmp_path):
    """Feed camera positions that are exactly the GPS ENU scaled by a known
    factor + rotated + translated; the Umeyama fit must recover the scale."""
    import json

    from telemetry.parsers import synthesize_track
    from reconstruction.georef import georeference_from_cameras

    track = synthesize_track(n=20, pattern="orbit")
    enu = np.array(track.enu)

    # Build synthetic cameras = enu / known_scale (recon units), so
    # metres_per_unit should come back ~= known_scale.
    known_scale = 4.0
    recon = enu / known_scale
    cams = [{"image": f"frame_{i:06d}.jpg", "position": recon[i].tolist()}
            for i in range(len(recon))]
    cams_path = tmp_path / "cameras.json"
    cams_path.write_text(json.dumps(cams))

    res = georeference_from_cameras(str(cams_path), track, allow_synthetic=True)
    assert res.ok
    assert res.scale_metres_per_unit == pytest.approx(known_scale, rel=0.05)
    assert res.rms_error_m < 1.0  # near-perfect alignment


def test_confidence_report_on_synthetic_cloud():
    from reconstruction.confidence import analyze_reconstruction

    # A cube of points with one dense corner and one sparse corner.
    rng = np.random.default_rng(0)
    dense = rng.normal(0, 0.2, size=(2000, 3))
    sparse = rng.normal(10, 0.2, size=(50, 3))
    xyz = np.vstack([dense, sparse])

    # Write a tiny PLY-like cloud via open3d if available; else skip.
    o3d = pytest.importorskip("open3d")
    import tempfile, os

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    path = os.path.join(tempfile.gettempdir(), "conf_test.ply")
    o3d.io.write_point_cloud(path, pcd)

    rep = analyze_reconstruction(point_cloud_path=path)
    assert rep.total_points == len(xyz)
    assert 0.0 <= rep.overall_confidence <= 1.0
    assert rep.high_conf_fraction + rep.medium_conf_fraction + rep.low_conf_fraction == pytest.approx(1.0, abs=0.01)
