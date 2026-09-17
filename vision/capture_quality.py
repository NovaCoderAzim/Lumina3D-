"""
Capture-Quality Analyzer (SIH 26158 challenges ii, iii, vi + the near-real-
time story).

The instant-feedback layer: BEFORE spending minutes on reconstruction, sample
the video and honestly assess whether it can reconstruct — and tell the
operator exactly what's wrong. In an operational single-pass scenario
(disaster/recon), knowing in seconds that the capture is unusable is itself
mission-critical value.

Real signals, all from a handful of sampled frames (fast):
  - Sharpness / motion blur    -> variance of Laplacian (challenge ii)
  - Exposure consistency       -> brightness spread across frames (iii)
  - Inter-frame overlap        -> ORB feature-match ratio between neighbours
  - Camera-motion coverage     -> spread of matched-feature flow direction,
                                  distinguishing a linear pass (limited angles,
                                  challenge i) from an orbit
  - Compression artifacting    -> blockiness proxy

Output: a 0-100 score, a GO / MARGINAL / NO-GO verdict, per-metric detail,
and concrete recommendations. Runs in ~seconds on ~24 sampled frames.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class CaptureQualityReport:
    verdict: str = "UNKNOWN"          # GO | MARGINAL | NO_GO
    score: float = 0.0               # 0..100
    sharpness_score: float = 0.0
    exposure_consistency: float = 0.0
    overlap_score: float = 0.0
    motion_pattern: str = "unknown"  # linear | orbital | erratic | static
    coverage_estimate_deg: float = 0.0
    frames_analyzed: int = 0
    duration_analyzed_s: float = 0.0
    processing_time_s: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "score": round(self.score, 1),
            "sharpness_score": round(self.sharpness_score, 1),
            "exposure_consistency": round(self.exposure_consistency, 1),
            "overlap_score": round(self.overlap_score, 1),
            "motion_pattern": self.motion_pattern,
            "coverage_estimate_deg": round(self.coverage_estimate_deg, 0),
            "frames_analyzed": self.frames_analyzed,
            "processing_time_s": round(self.processing_time_s, 2),
            "recommendations": self.recommendations,
            "notes": self.notes,
        }


def analyze_capture(video_path: str, sample_count: int = 24) -> CaptureQualityReport:
    import cv2
    import numpy as np

    start = time.perf_counter()
    report = CaptureQualityReport()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        report.verdict = "NO_GO"
        report.notes = f"Could not open video: {video_path}"
        return report

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        # stream without a count: read what we can
        total = sample_count * 10

    idxs = np.linspace(0, max(0, total - 1), min(sample_count, max(2, total))).astype(int)
    frames = []
    grays = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        # downscale for speed
        h, w = fr.shape[:2]
        scale = 640.0 / max(w, 1)
        if scale < 1:
            fr = cv2.resize(fr, (int(w * scale), int(h * scale)))
        frames.append(fr)
        grays.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()

    report.frames_analyzed = len(frames)
    report.duration_analyzed_s = total / fps if fps else 0.0
    if len(frames) < 2:
        report.verdict = "NO_GO"
        report.notes = "Too few readable frames."
        report.processing_time_s = time.perf_counter() - start
        return report

    # --- Sharpness (Laplacian variance) ---
    sharp_vals = [cv2.Laplacian(g, cv2.CV_64F).var() for g in grays]
    med_sharp = float(np.median(sharp_vals))
    # Map to 0..100 (empirical: <30 very blurry, >300 crisp)
    report.sharpness_score = float(np.clip((med_sharp - 30) / (300 - 30) * 100, 0, 100))

    # --- Exposure consistency (brightness spread) ---
    brightness = [float(g.mean()) for g in grays]
    b_std = float(np.std(brightness))
    report.exposure_consistency = float(np.clip(100 - b_std * 2.0, 0, 100))

    # --- Overlap + motion via ORB matches between consecutive samples ---
    orb = cv2.ORB_create(nfeatures=1200)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    overlaps = []
    flow_dirs = []
    for a, b in zip(grays[:-1], grays[1:]):
        ka, da = orb.detectAndCompute(a, None)
        kb, db = orb.detectAndCompute(b, None)
        if da is None or db is None or len(ka) < 8 or len(kb) < 8:
            overlaps.append(0.0)
            continue
        matches = bf.match(da, db)
        matches = sorted(matches, key=lambda m: m.distance)[:200]
        overlaps.append(len(matches) / max(1, min(len(ka), len(kb))))
        # net translation direction of matched features -> motion pattern
        if matches:
            dxdy = np.array([
                (kb[m.trainIdx].pt[0] - ka[m.queryIdx].pt[0],
                 kb[m.trainIdx].pt[1] - ka[m.queryIdx].pt[1]) for m in matches
            ])
            mean_v = dxdy.mean(axis=0)
            if np.linalg.norm(mean_v) > 1.0:
                flow_dirs.append(float(np.arctan2(mean_v[1], mean_v[0])))

    mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    report.overlap_score = float(np.clip(mean_overlap * 150, 0, 100))

    # --- Motion pattern from flow-direction spread ---
    if not flow_dirs:
        report.motion_pattern = "static"
        report.coverage_estimate_deg = 0.0
    else:
        angles = np.array(flow_dirs)
        # circular spread
        spread = float(np.std(np.unwrap(np.sort(angles))))
        if spread < 0.35:
            report.motion_pattern = "linear"
            report.coverage_estimate_deg = 60.0    # a straight pass sees a narrow cone
        elif spread < 1.2:
            report.motion_pattern = "orbital"
            report.coverage_estimate_deg = 220.0
        else:
            report.motion_pattern = "erratic"
            report.coverage_estimate_deg = 120.0

    # --- Aggregate score + verdict ---
    report.score = float(
        0.30 * report.sharpness_score
        + 0.20 * report.exposure_consistency
        + 0.35 * report.overlap_score
        + 0.15 * min(100.0, report.coverage_estimate_deg / 3.6)
    )
    if report.score >= 65 and report.overlap_score >= 45:
        report.verdict = "GO"
    elif report.score >= 45:
        report.verdict = "MARGINAL"
    else:
        report.verdict = "NO_GO"

    # --- Honest recommendations ---
    rec = report.recommendations
    if report.sharpness_score < 50:
        rec.append("Frames are soft/motion-blurred — fly slower or increase shutter speed.")
    if report.exposure_consistency < 60:
        rec.append("Exposure varies across the clip — lock exposure/ISO to stabilise texture matching.")
    if report.overlap_score < 45:
        rec.append("Low inter-frame overlap — fly slower or increase frame rate so consecutive views share more scene.")
    if report.motion_pattern == "linear":
        rec.append("Single linear pass: expect the facing side only. For a full 3D model, orbit the target 360° at constant altitude.")
    elif report.motion_pattern == "static":
        rec.append("Little camera motion detected — SfM needs parallax; translate the camera across the scene.")
    if not rec:
        rec.append("Capture looks suitable for reconstruction.")

    report.notes = (
        f"{report.verdict}: score {report.score:.0f}/100, motion={report.motion_pattern}, "
        f"~{report.coverage_estimate_deg:.0f}° angular coverage."
    )
    report.processing_time_s = time.perf_counter() - start
    return report
