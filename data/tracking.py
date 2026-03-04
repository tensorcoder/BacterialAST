"""IoU-based SORT tracker for linking bacteria detections across frames.

Implements a Simple Online and Realtime Tracking (SORT) variant that uses
oriented-bounding-box (OBB) IoU computed via Shapely polygon intersection.
Hungarian assignment is solved with ``lap.lapjv`` for speed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import lap
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OBB IoU helpers
# ---------------------------------------------------------------------------

def obb_to_polygon(cx: float, cy: float, w: float, h: float, angle_deg: float) -> Polygon:
    """Convert an oriented bounding box to a :class:`Shapely.Polygon`.

    Parameters
    ----------
    cx, cy : float
        Centre coordinates.
    w, h : float
        Width and height of the box (before rotation).
    angle_deg : float
        Counter-clockwise rotation in degrees.

    Returns
    -------
    Polygon
        A four-vertex polygon representing the OBB.
    """
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Half extents along the box-local axes.
    hw, hh = w / 2.0, h / 2.0

    # Corner offsets in box-local frame (before rotation).
    dx = np.array([-hw, hw, hw, -hw])
    dy = np.array([-hh, -hh, hh, hh])

    # Rotate offsets.
    xs = cx + dx * cos_a - dy * sin_a
    ys = cy + dx * sin_a + dy * cos_a

    return Polygon(zip(xs, ys))


def obb_iou(
    box_a: tuple[float, float, float, float, float],
    box_b: tuple[float, float, float, float, float],
) -> float:
    """Compute IoU between two oriented bounding boxes.

    Each box is ``(cx, cy, w, h, angle_deg)``.

    Returns
    -------
    float
        Intersection-over-union in [0, 1].
    """
    poly_a = obb_to_polygon(*box_a)
    poly_b = obb_to_polygon(*box_b)

    if not poly_a.is_valid or not poly_b.is_valid:
        return 0.0

    inter = poly_a.intersection(poly_b).area
    union = poly_a.area + poly_b.area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def compute_iou_matrix(
    tracks: list[_Track],
    detections: list[_Detection],
) -> NDArray[np.float64]:
    """Build a cost matrix of (1 - IoU) between tracks and detections.

    Parameters
    ----------
    tracks : list[_Track]
        Active tracks (rows).
    detections : list[_Detection]
        Current-frame detections (columns).

    Returns
    -------
    ndarray
        Shape ``(len(tracks), len(detections))``, values in [0, 1].
    """
    n_tracks = len(tracks)
    n_dets = len(detections)
    cost = np.ones((n_tracks, n_dets), dtype=np.float64)

    for i, trk in enumerate(tracks):
        box_a = trk.obb_tuple()
        for j, det in enumerate(detections):
            box_b = det.obb_tuple()
            cost[i, j] = 1.0 - obb_iou(box_a, box_b)
    return cost


# ---------------------------------------------------------------------------
# Internal track / detection representations
# ---------------------------------------------------------------------------

@dataclass
class _Detection:
    """Lightweight container for a single frame detection."""

    frame_idx: int
    detection_id: int
    cx: float
    cy: float
    w: float
    h: float
    angle: float
    confidence: float

    def obb_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.cx, self.cy, self.w, self.h, self.angle)


@dataclass
class _Track:
    """Internal mutable state for one tracked object."""

    track_id: int
    cx: float
    cy: float
    w: float
    h: float
    angle: float
    confidence: float
    age: int = 0  # frames since last matched detection
    total_hits: int = 1  # number of frames with a matched detection
    frame_start: int = 0
    frame_end: int = 0
    history: list[_Detection] = field(default_factory=list)

    def obb_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.cx, self.cy, self.w, self.h, self.angle)

    def update(self, det: _Detection) -> None:
        """Update track state with a matched detection."""
        self.cx = det.cx
        self.cy = det.cy
        self.w = det.w
        self.h = det.h
        self.angle = det.angle
        self.confidence = det.confidence
        self.age = 0
        self.total_hits += 1
        self.frame_end = det.frame_idx
        self.history.append(det)


# ---------------------------------------------------------------------------
# Track summary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackSummary:
    """Per-track statistics returned after tracking is complete."""

    track_id: int
    frame_start: int
    frame_end: int
    total_hits: int
    track_length: int  # frame_end - frame_start + 1
    mean_confidence: float
    mean_cx: float
    mean_cy: float


# ---------------------------------------------------------------------------
# SORT-style tracker
# ---------------------------------------------------------------------------

class BacteriaTracker:
    """IoU-based SORT tracker for oriented bounding boxes.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU for a valid track-detection match (default 0.3).
    max_age : int
        Maximum number of consecutive frames a track may remain unmatched
        before deletion (default 15).
    min_hits : int
        Minimum number of matched detections before a track is considered
        *confirmed* and eligible for output (default 5).
    min_track_length : int
        Minimum track duration in frames (``frame_end - frame_start + 1``)
        to be kept in the final output (default 150).
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 15,
        min_hits: int = 5,
        min_track_length: int = 150,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.min_track_length = min_track_length

        self._next_id: int = 0
        self._active_tracks: list[_Track] = []
        self._finished_tracks: list[_Track] = []

    def _allocate_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    # -- Hungarian matching ---------------------------------------------------

    def _match(
        self,
        tracks: list[_Track],
        detections: list[_Detection],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Solve the assignment problem.

        Returns
        -------
        matched : list of (track_index, detection_index)
        unmatched_tracks : list of track indices
        unmatched_detections : list of detection indices
        """
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        if len(detections) == 0:
            return [], list(range(len(tracks))), []

        cost = compute_iou_matrix(tracks, detections)

        # lap.lapjv expects a square or rectangular cost matrix.
        # It returns (min_cost, row_to_col, col_to_row).
        _, row_to_col, _ = lap.lapjv(cost, extend_cost=True, cost_limit=1.0 - self.iou_threshold)

        matched: list[tuple[int, int]] = []
        unmatched_tracks: list[int] = []
        unmatched_detections: set[int] = set(range(len(detections)))

        for trk_idx, det_idx in enumerate(row_to_col):
            if det_idx < 0:
                # No valid assignment for this track.
                unmatched_tracks.append(trk_idx)
            else:
                matched.append((trk_idx, det_idx))
                unmatched_detections.discard(det_idx)

        return matched, unmatched_tracks, sorted(unmatched_detections)

    # -- Single-frame update --------------------------------------------------

    def _update_frame(self, frame_idx: int, detections: list[_Detection]) -> dict[int, int]:
        """Process detections for a single frame.

        Returns
        -------
        assignment : dict
            Mapping from ``detection_id`` to ``track_id`` for confirmed
            tracks only.
        """
        matched, unmatched_trk, unmatched_det = self._match(
            self._active_tracks, detections
        )

        # 1. Update matched tracks.
        for trk_idx, det_idx in matched:
            self._active_tracks[trk_idx].update(detections[det_idx])

        # 2. Age unmatched tracks and retire old ones.
        tracks_to_remove: list[int] = []
        for trk_idx in unmatched_trk:
            self._active_tracks[trk_idx].age += 1
            if self._active_tracks[trk_idx].age > self.max_age:
                tracks_to_remove.append(trk_idx)

        # Remove in reverse order to keep indices valid.
        for idx in sorted(tracks_to_remove, reverse=True):
            self._finished_tracks.append(self._active_tracks.pop(idx))

        # 3. Create new tracks for unmatched detections.
        for det_idx in unmatched_det:
            det = detections[det_idx]
            new_track = _Track(
                track_id=self._allocate_id(),
                cx=det.cx,
                cy=det.cy,
                w=det.w,
                h=det.h,
                angle=det.angle,
                confidence=det.confidence,
                age=0,
                total_hits=1,
                frame_start=frame_idx,
                frame_end=frame_idx,
                history=[det],
            )
            self._active_tracks.append(new_track)

        # 4. Build assignment dict for this frame (confirmed tracks only).
        assignment: dict[int, int] = {}
        for trk_idx, det_idx in matched:
            trk = self._active_tracks[trk_idx]
            if trk.total_hits >= self.min_hits:
                assignment[detections[det_idx].detection_id] = trk.track_id

        # Also include newly created tracks that are already confirmed
        # (unlikely with min_hits > 1, but for correctness).
        for trk in self._active_tracks:
            if trk.age == 0 and trk.total_hits >= self.min_hits:
                if trk.history:
                    last_det = trk.history[-1]
                    if last_det.frame_idx == frame_idx:
                        assignment.setdefault(last_det.detection_id, trk.track_id)

        return assignment

    # -- Public API -----------------------------------------------------------

    def process_experiment(self, detections_df: pd.DataFrame) -> tuple[pd.DataFrame, list[TrackSummary]]:
        """Run tracking on an experiment's detections.

        Parameters
        ----------
        detections_df : DataFrame
            Must contain columns: ``frame_idx``, ``detection_id``, ``cx``,
            ``cy``, ``w``, ``h``, ``angle``, ``confidence``.

        Returns
        -------
        result_df : DataFrame
            A copy of *detections_df* with an added ``track_id`` column.
            Detections that do not belong to a confirmed, long-enough track
            have ``track_id = -1``.
        summaries : list[TrackSummary]
            Per-track statistics for all tracks passing the
            ``min_track_length`` filter.
        """
        required_cols = {"frame_idx", "detection_id", "cx", "cy", "w", "h", "angle", "confidence"}
        missing = required_cols - set(detections_df.columns)
        if missing:
            raise ValueError(f"Missing columns in detections_df: {missing}")

        # Reset internal state.
        self._next_id = 0
        self._active_tracks = []
        self._finished_tracks = []

        # Sort by frame to ensure temporal order.
        df = detections_df.sort_values("frame_idx").reset_index(drop=True)

        # Map (frame_idx, detection_id) -> track_id.
        track_assignments: dict[tuple[int, int], int] = {}

        grouped = df.groupby("frame_idx", sort=True)
        for frame_idx, group in grouped:
            dets = [
                _Detection(
                    frame_idx=int(row.frame_idx),
                    detection_id=int(row.detection_id),
                    cx=float(row.cx),
                    cy=float(row.cy),
                    w=float(row.w),
                    h=float(row.h),
                    angle=float(row.angle),
                    confidence=float(row.confidence),
                )
                for row in group.itertuples(index=False)
            ]
            assignment = self._update_frame(int(frame_idx), dets)
            for det_id, trk_id in assignment.items():
                track_assignments[(int(frame_idx), det_id)] = trk_id

        # Flush remaining active tracks to finished.
        self._finished_tracks.extend(self._active_tracks)
        self._active_tracks = []

        # Build track_id column.
        track_ids = []
        for row in df.itertuples(index=False):
            key = (int(row.frame_idx), int(row.detection_id))
            track_ids.append(track_assignments.get(key, -1))

        result_df = df.copy()
        result_df["track_id"] = track_ids

        # -- Build summaries and apply min_track_length filter ----------------
        all_tracks = self._finished_tracks
        summaries: list[TrackSummary] = []

        # Also collect which track_ids pass the length filter.
        valid_track_ids: set[int] = set()

        for trk in all_tracks:
            if trk.total_hits < self.min_hits:
                continue
            track_length = trk.frame_end - trk.frame_start + 1
            if track_length < self.min_track_length:
                continue

            valid_track_ids.add(trk.track_id)

            confs = [d.confidence for d in trk.history]
            cxs = [d.cx for d in trk.history]
            cys = [d.cy for d in trk.history]

            summaries.append(
                TrackSummary(
                    track_id=trk.track_id,
                    frame_start=trk.frame_start,
                    frame_end=trk.frame_end,
                    total_hits=trk.total_hits,
                    track_length=track_length,
                    mean_confidence=float(np.mean(confs)),
                    mean_cx=float(np.mean(cxs)),
                    mean_cy=float(np.mean(cys)),
                )
            )

        # Set track_id = -1 for detections belonging to tracks that did not
        # pass the min_track_length filter.
        result_df.loc[~result_df["track_id"].isin(valid_track_ids), "track_id"] = -1

        logger.info(
            "Tracking complete: %d total tracks, %d passed filters (min_hits=%d, min_length=%d)",
            len(all_tracks),
            len(summaries),
            self.min_hits,
            self.min_track_length,
        )

        return result_df, summaries
