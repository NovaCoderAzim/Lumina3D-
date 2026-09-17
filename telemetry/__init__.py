"""Lumina3D telemetry module: GPS / flight-metadata ingestion + geodesy.

Underpins metric accuracy and georeferencing without GCPs.
"""

from .interface import TelemetryLoader
from .schemas import FlightTrack, GpsFix

__all__ = ["TelemetryLoader", "FlightTrack", "GpsFix"]
