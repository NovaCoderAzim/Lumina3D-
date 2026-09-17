"""
Sensor and hardware capability detector (Section 15 & 16).

Discovers available sensor modalities:
- Video (always required)
- GPS / GNSS
- IMU / Flight Controller
- LiDAR
- RTK / PPK
- Factory / Exif Camera Intrinsics
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SensorCapabilities:
    video: bool = True
    gps: bool = False
    imu: bool = False
    lidar: bool = False
    rtk: bool = False
    ppk: bool = False
    camera_intrinsics: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video": self.video,
            "gps": self.gps,
            "imu": self.imu,
            "lidar": self.lidar,
            "rtk": self.rtk,
            "ppk": self.ppk,
            "camera_intrinsics": self.camera_intrinsics,
            "details": self.details,
        }


def detect_capabilities(
    video_path: Optional[str] = None,
    project_dir: Optional[Path] = None,
) -> SensorCapabilities:
    """Discovers which real sensor data feeds are available for a given project."""
    caps = SensorCapabilities()

    if not video_path:
        caps.video = False
        return caps

    vp = Path(video_path)
    if not vp.exists():
        caps.video = False
        return caps

    caps.video = True
    caps.details["video_format"] = vp.suffix.lower()

    # Search for GPS sidecars (.srt, .csv, .gps)
    parent_dir = vp.parent if vp.parent.exists() else None
    if parent_dir:
        for ext in (".srt", ".SRT"):
            if vp.with_suffix(ext).exists():
                caps.gps = True
                caps.details["gps_source"] = f"dji_srt:{vp.with_suffix(ext).name}"
                break
        if not caps.gps:
            for ext in (".csv", ".gps", ".txt"):
                if vp.with_suffix(ext).exists():
                    caps.gps = True
                    caps.details["gps_source"] = f"csv_log:{vp.with_suffix(ext).name}"
                    break

    # Look for IMU, LiDAR, RTK/PPK folders or sidecars
    if project_dir and project_dir.exists():
        lidar_dir = project_dir / "lidar"
        if lidar_dir.exists() and any(lidar_dir.iterdir()):
            caps.lidar = True
            caps.details["lidar_files"] = [p.name for p in lidar_dir.glob("*.las")] + [p.name for p in lidar_dir.glob("*.laz")]

        rtk_file = project_dir / "rtk.obs"
        if rtk_file.exists():
            caps.rtk = True
            caps.details["rtk_file"] = rtk_file.name

    return caps


# --------------------------------------------------------------------------
# Clean Sensor Fusion Interfaces (Section 16 - Future SIH Extensibility)
# --------------------------------------------------------------------------

class GPSProvider(ABC):
    @abstractmethod
    def get_track(self) -> Any:
        raise NotImplementedError


class IMUProvider(ABC):
    @abstractmethod
    def get_orientations(self) -> List[Any]:
        raise NotImplementedError


class LiDARProvider(ABC):
    @abstractmethod
    def get_point_cloud(self) -> Any:
        raise NotImplementedError


class RTKProvider(ABC):
    @abstractmethod
    def get_corrected_positions(self) -> List[Any]:
        raise NotImplementedError
