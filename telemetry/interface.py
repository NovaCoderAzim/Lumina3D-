"""
Public entry point for the telemetry module.

    from telemetry import TelemetryLoader
    track = TelemetryLoader().load_for_video(video_path, num_selected_frames=60)

Auto-detects a GPS source next to the video (DJI .SRT / .srt, or a .csv/.gps
sidecar). Falls back to a clearly-flagged synthetic track so the metric
pipeline is always exercisable. Maps GPS fixes onto the selected frame
indices by time so reconstruction can align cameras to GPS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .parsers import parse_dji_srt, parse_gps_csv, synthesize_track, post_process
from .schemas import FlightTrack


class TelemetryLoader:
    def load_for_video(
        self,
        video_path: Optional[str],
        num_selected_frames: int = 60,
        synth_pattern: str = "linear",
        allow_synthetic: bool = False,
    ) -> FlightTrack:
        track = self._auto_detect(video_path)
        if track is None or not track.has_data():
            if allow_synthetic:
                track = synthesize_track(n=max(2, num_selected_frames), pattern=synth_pattern)
            else:
                return FlightTrack(source="none", fixes=[], notes="No GPS telemetry sidecar provided. Video-only processing.")
        track = self._map_to_frames(track, num_selected_frames)
        return track

    def load_file(self, path: str) -> FlightTrack:
        p = Path(path)
        if p.suffix.lower() == ".srt":
            return parse_dji_srt(p)
        return parse_gps_csv(p)

    # ------------------------------------------------------------------

    def _auto_detect(self, video_path: Optional[str]) -> Optional[FlightTrack]:
        if not video_path:
            return None
        vp = Path(video_path)
        if not vp.exists():
            return None
        # Look for a sidecar with the same stem, common extensions.
        for ext in (".srt", ".SRT"):
            cand = vp.with_suffix(ext)
            if cand.exists():
                return parse_dji_srt(cand)
        for ext in (".csv", ".gps", ".txt"):
            cand = vp.with_suffix(ext)
            if cand.exists():
                return parse_gps_csv(cand)
        # Also look in the same directory for any *.srt / *.csv.
        for pattern in ("*.srt", "*.SRT", "*.csv"):
            for cand in vp.parent.glob(pattern):
                if cand.suffix.lower() == ".srt":
                    return parse_dji_srt(cand)
                return parse_gps_csv(cand)
        return None

    def _map_to_frames(self, track: FlightTrack, num_frames: int) -> FlightTrack:
        """Assign each selected frame index a fix by resampling the track
        uniformly across its fixes (frames are uniformly sampled in time)."""
        if not track.fixes or num_frames <= 0:
            return track
        n = len(track.fixes)
        for k in range(num_frames):
            # nearest fix for this frame
            idx = min(n - 1, round(k * (n - 1) / max(1, num_frames - 1)))
            track.fixes[idx].frame_index = k
        return post_process(track)
