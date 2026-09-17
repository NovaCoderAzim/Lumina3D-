"""DRISHTI-3D telemetry module: GPS / flight-metadata ingestion + geodesy.

Solves SIH 26158 mandatory input (GPS + flight metadata) and underpins
metric accuracy without GCPs (challenge viii).
"""

from .interface import TelemetryLoader
from .schemas import FlightTrack, GpsFix

__all__ = ["TelemetryLoader", "FlightTrack", "GpsFix"]
