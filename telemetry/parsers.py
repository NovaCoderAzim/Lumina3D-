"""
Telemetry parsers.

Real drone footage carries GPS/flight metadata in a few standard forms:
  - DJI .SRT sidecar (the most common — a subtitle file next to the .MP4
    with per-frame GPS/altitude/gimbal fields).
  - A GPS CSV log (many flight-log exporters and Litchi/Airdata produce this).

When none is present, `synthesize_track` generates a physically-plausible
single-pass flight track so the metric-scaling pipeline is exercisable and
ready for real telemetry. Synthetic tracks are clearly flagged
(is_synthetic=True) and never presented as real GPS.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Optional

from .schemas import FlightTrack, GpsFix, ecef_to_enu, geodetic_to_ecef, haversine_m

# DJI SRT fields vary by firmware; these cover the common variants.
_LAT_RE = re.compile(r"\[?latitude\s*[:=]\s*([-\d.]+)\]?", re.I)
_LON_RE = re.compile(r"\[?long?itude\s*[:=]\s*([-\d.]+)\]?", re.I)
_ABS_ALT_RE = re.compile(r"abs_alt\s*[:=]\s*([-\d.]+)", re.I)
_REL_ALT_RE = re.compile(r"rel_alt\s*[:=]\s*([-\d.]+)", re.I)
_GPS_TUPLE_RE = re.compile(r"GPS\s*\(?\s*([-\d.]+)[,\s]+([-\d.]+)[,\s]+([-\d.]+)", re.I)
_TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->")


def _srt_time_to_seconds(block: str) -> Optional[float]:
    m = _TIME_RE.search(block)
    if not m:
        return None
    h, mm, s, ms = (int(x) for x in m.groups())
    return h * 3600 + mm * 60 + s + ms / 1000.0


def parse_dji_srt(srt_path: str | Path) -> FlightTrack:
    text = Path(srt_path).read_text(errors="ignore")
    blocks = re.split(r"\n\s*\n", text)
    fixes: List[GpsFix] = []
    for i, block in enumerate(blocks):
        if not block.strip():
            continue
        lat = lon = alt = None
        rel = None
        gt = _GPS_TUPLE_RE.search(block)
        if gt:
            lon, lat, alt = float(gt.group(1)), float(gt.group(2)), float(gt.group(3))
            # DJI GPS() is usually (lon, lat, alt); some are (lat, lon). Guess
            # by magnitude: |lat|<=90.
            if abs(lon) <= 90 and abs(lat) > 90:
                lat, lon = lon, lat
        else:
            ml, mo = _LAT_RE.search(block), _LON_RE.search(block)
            if ml and mo:
                lat, lon = float(ml.group(1)), float(mo.group(1))
            ma = _ABS_ALT_RE.search(block)
            if ma:
                alt = float(ma.group(1))
        mr = _REL_ALT_RE.search(block)
        if mr:
            rel = float(mr.group(1))
        if lat is None or lon is None:
            continue
        t = _srt_time_to_seconds(block)
        if t is None:
            t = float(i)  # fall back to block index as pseudo-seconds
        fixes.append(
            GpsFix(t_seconds=t, latitude=lat, longitude=lon,
                   altitude_m=alt if alt is not None else 0.0, rel_altitude_m=rel)
        )
    track = FlightTrack(source="dji_srt", fixes=fixes,
                        notes=f"Parsed {len(fixes)} GPS fixes from DJI SRT.")
    return post_process(track)


def parse_gps_csv(csv_path: str | Path) -> FlightTrack:
    import csv as _csv

    fixes: List[GpsFix] = []
    with open(csv_path, newline="", errors="ignore") as fh:
        reader = _csv.DictReader(fh)
        # Normalise header names.
        def col(row, *names):
            for n in names:
                for k in row:
                    if k and k.strip().lower() == n:
                        return row[k]
            return None

        for i, row in enumerate(reader):
            lat = col(row, "latitude", "lat", "gps_lat")
            lon = col(row, "longitude", "lon", "lng", "gps_lon")
            alt = col(row, "altitude", "alt", "altitude_m", "abs_alt")
            t = col(row, "time", "t", "seconds", "timestamp")
            if lat is None or lon is None:
                continue
            try:
                fixes.append(
                    GpsFix(
                        t_seconds=float(t) if t not in (None, "") else float(i),
                        latitude=float(lat),
                        longitude=float(lon),
                        altitude_m=float(alt) if alt not in (None, "") else 0.0,
                    )
                )
            except ValueError:
                continue
    track = FlightTrack(source="csv", fixes=fixes,
                        notes=f"Parsed {len(fixes)} GPS fixes from CSV.")
    return post_process(track)


def synthesize_track(
    n: int = 60,
    duration_s: float = 30.0,
    anchor_lat: float = 27.1751,      # Taj Mahal, as a realistic default
    anchor_lon: float = 78.0421,
    base_alt: float = 100.0,
    pattern: str = "linear",
) -> FlightTrack:
    """Generate a physically-plausible single-pass track. Clearly flagged
    synthetic. `pattern`: 'linear' (single straight pass, the PS's core
    constraint) or 'orbit'."""
    fixes: List[GpsFix] = []
    # metres-per-degree at this latitude
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(anchor_lat))

    for i in range(n):
        f = i / max(1, n - 1)
        t = f * duration_s
        if pattern == "orbit":
            ang = 2 * math.pi * f
            radius_m = 120.0
            east = radius_m * math.cos(ang)
            north = radius_m * math.sin(ang)
            alt = base_alt + 8.0 * math.sin(ang * 1.5)
        else:  # linear single pass
            east = -150.0 + 300.0 * f      # fly 300 m west->east
            north = 40.0 * f               # slight drift
            alt = base_alt + 5.0 * math.sin(f * math.pi)
        lat = anchor_lat + north / m_per_deg_lat
        lon = anchor_lon + east / m_per_deg_lon
        fixes.append(
            GpsFix(t_seconds=t, latitude=lat, longitude=lon, altitude_m=alt,
                   rel_altitude_m=alt - base_alt)
        )
    track = FlightTrack(
        source="synthetic", fixes=fixes, is_synthetic=True,
        notes=(f"SYNTHETIC {pattern} track ({n} fixes) — no real telemetry "
               "was provided. Metric scale is illustrative until real GPS is "
               "supplied."),
    )
    return post_process(track)


def post_process(track: FlightTrack) -> FlightTrack:
    """Fill derived geometry: local ENU positions, total length, duration."""
    if not track.fixes:
        return track
    a = track.fixes[0]
    track.anchor_lat, track.anchor_lon, track.anchor_alt = a.latitude, a.longitude, a.altitude_m
    enu: List[List[float]] = []
    length = 0.0
    prev = None
    for fx in track.fixes:
        x, y, z = geodetic_to_ecef(fx.latitude, fx.longitude, fx.altitude_m)
        e = ecef_to_enu(x, y, z, a.latitude, a.longitude, a.altitude_m)
        enu.append(e)
        if prev is not None:
            length += haversine_m(prev.latitude, prev.longitude, fx.latitude, fx.longitude)
        prev = fx
    track.enu = enu
    track.total_length_m = length
    track.duration_s = track.fixes[-1].t_seconds - track.fixes[0].t_seconds
    return track
