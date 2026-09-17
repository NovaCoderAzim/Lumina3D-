"""
Telemetry schemas — GPS / flight-metadata contract.

Dependency-free stdlib dataclasses so any module can consume a flight
track without pulling in parsers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GpsFix:
    """One GPS/telemetry sample tied to a video timestamp (and, once frames
    are selected, to a frame index)."""

    t_seconds: float          # time offset from start of video
    latitude: float           # WGS84 degrees
    longitude: float          # WGS84 degrees
    altitude_m: float         # metres (ellipsoidal or barometric)
    frame_index: Optional[int] = None
    # Optional richer telemetry.
    rel_altitude_m: Optional[float] = None   # barometric relative altitude
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t_seconds": round(self.t_seconds, 3),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude_m": self.altitude_m,
            "frame_index": self.frame_index,
            "rel_altitude_m": self.rel_altitude_m,
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
        }


@dataclass
class FlightTrack:
    """The full parsed flight path for one video."""

    source: str               # "dji_srt" | "csv" | "synthetic" | "none"
    fixes: List[GpsFix] = field(default_factory=list)
    is_synthetic: bool = False
    notes: str = ""

    # ---- derived geometry (filled by post_process) ----
    total_length_m: float = 0.0
    duration_s: float = 0.0
    # Local ENU (East-North-Up, metres) positions per fix, anchored at the
    # first fix. This is what we align the SfM reconstruction against.
    enu: List[List[float]] = field(default_factory=list)
    anchor_lat: Optional[float] = None
    anchor_lon: Optional[float] = None
    anchor_alt: Optional[float] = None

    def has_data(self) -> bool:
        return len(self.fixes) >= 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "is_synthetic": self.is_synthetic,
            "notes": self.notes,
            "num_fixes": len(self.fixes),
            "total_length_m": round(self.total_length_m, 2),
            "duration_s": round(self.duration_s, 2),
            "anchor": {
                "lat": self.anchor_lat,
                "lon": self.anchor_lon,
                "alt": self.anchor_alt,
            },
            "fixes": [f.to_dict() for f in self.fixes],
        }


# ---------------------------------------------------------------------------
# Geodesy helpers (WGS84 -> local ENU metres). No external deps.
# ---------------------------------------------------------------------------

_WGS84_A = 6378137.0            # semi-major axis (m)
_WGS84_F = 1.0 / 298.257223563  # flattening
_WGS84_E2 = _WGS84_F * (2 - _WGS84_F)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    n = _WGS84_A / math.sqrt(1 - _WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * math.cos(lon)
    y = (n + alt_m) * cos_lat * math.sin(lon)
    z = (n * (1 - _WGS84_E2) + alt_m) * sin_lat
    return x, y, z


def ecef_to_enu(x, y, z, lat0_deg, lon0_deg, alt0):
    """Convert an ECEF point to local East-North-Up around an anchor."""
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0)
    dx, dy, dz = x - x0, y - y0, z - z0
    sin_lat, cos_lat = math.sin(lat0), math.cos(lat0)
    sin_lon, cos_lon = math.sin(lon0), math.cos(lon0)
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return [east, north, up]


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
