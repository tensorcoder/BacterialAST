"""YOLO-OBB detection, oriented crop extraction, and HDF5 storage.

Processes raw BMP microscopy frames through YOLOv11-OBB to detect bacteria,
filters to only in-focus detections, rectifies oriented bounding box crops
via affine transform, resizes to 96x96 grayscale, and persists them in
per-experiment HDF5 files with structured metadata including timestamps.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import h5py
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from ultralytics import YOLO

logger = logging.getLogger(__name__)

CROP_SIZE: int = 128

# Regex to parse image filenames: image_{timestamp}[_{MIC}].{bmp,tiff,tif}
_FILENAME_RE = re.compile(
    r"^image_(\d+(?:\.\d+)?)(?:_.+)?\.(?:bmp|tiff?)$", re.IGNORECASE
)


def parse_timestamp_from_filename(filename: str) -> float | None:
    """Extract unix timestamp from an image filename.

    Expected format: ``image_{unix_timestamp.ms}[_{MIC}].bmp``

    Examples::

        image_1741018345.67383_4mgL.bmp  →  1741018345.67383
        image_1741018345.67383.bmp       →  1741018345.67383

    Returns ``None`` if the filename does not match the expected pattern.
    """
    m = _FILENAME_RE.match(filename)
    if m is None:
        return None
    return float(m.group(1))


# ---------------------------------------------------------------------------
# Dataclass for a single oriented-bounding-box detection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OBBDetection:
    """One oriented-bounding-box detection."""

    detection_id: int
    cx: float
    cy: float
    w: float
    h: float
    angle: float  # degrees, counter-clockwise from x-axis
    confidence: float
    class_id: int = 0
    class_name: str = ""


# ---------------------------------------------------------------------------
# Structured NumPy dtype mirroring detection metadata (used inside HDF5)
# ---------------------------------------------------------------------------
METADATA_DTYPE = np.dtype(
    [
        ("timestamp", np.float64),
        ("detection_id", np.int32),
        ("cx", np.float32),
        ("cy", np.float32),
        ("w", np.float32),
        ("h", np.float32),
        ("angle", np.float32),
        ("confidence", np.float32),
    ]
)


# ---------------------------------------------------------------------------
# YOLO crop extractor
# ---------------------------------------------------------------------------
class YOLOCropExtractor:
    """Run YOLOv11-OBB on frames and extract rectified grayscale crops.

    Only detections classified as ``focused_class_name`` are kept.

    Parameters
    ----------
    model_path : str | Path
        Path to the YOLO-OBB weights file.
    crop_size : int
        Target side length of the square crop (default 96).
    batch_size : int
        Number of frames processed per YOLO inference call.
    conf_threshold : float
        Minimum detection confidence to keep.
    focused_class_name : str
        YOLO class name for in-focus bacteria (default ``"Focused"``).
    device : str | None
        Device string forwarded to YOLO (e.g. ``"cuda:0"``, ``"cpu"``).
        *None* lets Ultralytics choose automatically.
    """

    def __init__(
        self,
        model_path: str | Path = "/path/to/yolo11-obb.pt",
        crop_size: int = CROP_SIZE,
        batch_size: int = 16,
        conf_threshold: float = 0.25,
        focused_class_name: str = "Focused",
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.focused_class_name = focused_class_name
        self.device = device

        self._model: YOLO | None = None
        self._focused_class_id: int | None = None

    # -- lazy model loading ---------------------------------------------------

    @property
    def model(self) -> YOLO:
        """Lazily load the YOLO model on first access."""
        if self._model is None:
            logger.info("Loading YOLO-OBB model from %s", self.model_path)
            self._model = YOLO(str(self.model_path))
            # Resolve focused class ID from model names
            names = self._model.names  # {int: str}
            for cls_id, cls_name in names.items():
                if cls_name.lower() == self.focused_class_name.lower():
                    self._focused_class_id = cls_id
                    break
            if self._focused_class_id is None:
                raise ValueError(
                    f"Class '{self.focused_class_name}' not found in model. "
                    f"Available classes: {names}"
                )
            logger.info(
                "Focused class: '%s' (ID %d)",
                self.focused_class_name,
                self._focused_class_id,
            )
        return self._model

    # -- OBB crop extraction --------------------------------------------------

    @staticmethod
    def _rectify_obb_crop(
        image: NDArray[np.uint8],
        cx: float,
        cy: float,
        w: float,
        h: float,
        angle: float,
        crop_size: int,
    ) -> NDArray[np.uint8]:
        """Extract a size-preserving crop from an oriented bounding box.

        The region defined by (cx, cy, w, h, angle) is rectified using an
        affine warp so that the OBB becomes axis-aligned, then **centered
        on a fixed-size canvas** (``crop_size x crop_size``) using reflected
        border fill.  The bacterium is placed at its native pixel size so
        that both shape and absolute size are preserved.  Only the rare
        detections larger than ``crop_size`` are downscaled to fit.
        """
        rotation_matrix = cv2.getRotationMatrix2D(
            center=(cx, cy),
            angle=angle,
            scale=1.0,
        )

        img_h, img_w = image.shape[:2]
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (img_w, img_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        half_w, half_h = w / 2.0, h / 2.0
        x1 = max(int(round(cx - half_w)), 0)
        y1 = max(int(round(cy - half_h)), 0)
        x2 = min(int(round(cx + half_w)), img_w)
        y2 = min(int(round(cy + half_h)), img_h)

        crop = rotated[y1:y2, x1:x2]

        if crop.size == 0:
            return np.zeros((crop_size, crop_size), dtype=np.uint8)

        ch, cw = crop.shape[:2]

        # Downscale only if the crop exceeds the canvas size.
        if ch > crop_size or cw > crop_size:
            scale = crop_size / max(ch, cw)
            crop = cv2.resize(
                crop,
                (int(round(cw * scale)), int(round(ch * scale))),
                interpolation=cv2.INTER_LINEAR,
            )
            ch, cw = crop.shape[:2]

        # Centre the crop on a fixed-size canvas with reflected border.
        pad_top = (crop_size - ch) // 2
        pad_bot = crop_size - ch - pad_top
        pad_left = (crop_size - cw) // 2
        pad_right = crop_size - cw - pad_left
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bot, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )

        return crop

    def _parse_results(
        self,
        results: list,
        frame_indices: list[int],
    ) -> list[tuple[int, list[OBBDetection]]]:
        """Parse Ultralytics OBB results, keeping only focused detections."""
        # Ensure model is loaded (resolves _focused_class_id)
        _ = self.model

        parsed: list[tuple[int, list[OBBDetection]]] = []
        for frame_idx, result in zip(frame_indices, results):
            detections: list[OBBDetection] = []
            if result.obb is None:
                parsed.append((frame_idx, detections))
                continue

            obb_data = result.obb
            boxes = obb_data.xywhr.cpu().numpy()  # (N, 5)
            confs = obb_data.conf.cpu().numpy()  # (N,)
            classes = obb_data.cls.cpu().numpy().astype(int)  # (N,)

            for det_id, (box, conf, cls_id) in enumerate(
                zip(boxes, confs, classes)
            ):
                if conf < self.conf_threshold:
                    continue
                # Only keep focused detections
                if cls_id != self._focused_class_id:
                    continue

                cx, cy, w, h, angle_rad = box
                detections.append(
                    OBBDetection(
                        detection_id=det_id,
                        cx=float(cx),
                        cy=float(cy),
                        w=float(w),
                        h=float(h),
                        angle=float(np.degrees(angle_rad)),
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=self.focused_class_name,
                    )
                )
            parsed.append((frame_idx, detections))
        return parsed

    def detect_and_crop(
        self,
        image_paths: Sequence[str | Path],
        frame_indices: list[int],
        timestamps: list[float],
    ) -> tuple[list[NDArray[np.uint8]], list[np.void]]:
        """Run YOLO-OBB on a batch of images, returning focused crops and metadata.

        Parameters
        ----------
        image_paths : sequence of str or Path
            Paths to image files (YOLO handles loading internally).
        frame_indices : list[int]
            Frame indices corresponding to each image.
        timestamps : list[float]
            Unix timestamps corresponding to each image.

        Returns
        -------
        crops : list of ndarray
            Each element is a ``(crop_size, crop_size)`` uint8 array.
        metadata_rows : list of np.void
            Structured array rows matching ``METADATA_DTYPE``.
        """
        # Pass file paths to YOLO so it handles image loading correctly
        results = self.model.predict(
            source=[str(p) for p in image_paths],
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        parsed = self._parse_results(results, frame_indices)

        crops: list[NDArray[np.uint8]] = []
        metadata_rows: list[np.void] = []

        # Cache loaded grayscale images for crop extraction
        _image_cache: dict[int, NDArray[np.uint8]] = {}

        for frame_idx, detections in parsed:
            if not detections:
                continue

            src_idx = frame_indices.index(frame_idx)
            timestamp = timestamps[src_idx]

            # Load image as grayscale for crop extraction (only when needed)
            if src_idx not in _image_cache:
                img = cv2.imread(str(image_paths[src_idx]), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                _image_cache[src_idx] = img
            image = _image_cache[src_idx]

            for det in detections:
                crop = self._rectify_obb_crop(
                    image, det.cx, det.cy, det.w, det.h, det.angle, self.crop_size
                )
                crops.append(crop)
                row = np.array(
                    (
                        timestamp,
                        det.detection_id,
                        det.cx,
                        det.cy,
                        det.w,
                        det.h,
                        det.angle,
                        det.confidence,
                    ),
                    dtype=METADATA_DTYPE,
                )
                metadata_rows.append(row[()])
        return crops, metadata_rows


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------
class HDF5CropWriter:
    """Write extracted crops and metadata to an HDF5 file.

    The file layout is::

        /crops      (N, 96, 96)  uint8       -- resizable along axis 0
        /metadata   (N,)         structured   -- ``METADATA_DTYPE``
    """

    def __init__(
        self,
        crop_size: int = CROP_SIZE,
        chunk_size: int = 1024,
        compression: str | None = "gzip",
    ) -> None:
        self.crop_size = crop_size
        self.chunk_size = chunk_size
        self.compression = compression

    def create(self, path: str | Path) -> h5py.File:
        """Create (or overwrite) an HDF5 file with the expected datasets."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        f = h5py.File(str(path), "w")
        f.create_dataset(
            "crops",
            shape=(0, self.crop_size, self.crop_size),
            maxshape=(None, self.crop_size, self.crop_size),
            dtype=np.uint8,
            chunks=(min(self.chunk_size, 1), self.crop_size, self.crop_size),
            compression=self.compression,
        )
        f.create_dataset(
            "metadata",
            shape=(0,),
            maxshape=(None,),
            dtype=METADATA_DTYPE,
            chunks=(min(self.chunk_size, 1),),
            compression=self.compression,
        )
        return f

    @staticmethod
    def append(
        h5file: h5py.File,
        crops: list[NDArray[np.uint8]],
        metadata_rows: list[np.void],
    ) -> None:
        """Append a batch of crops and metadata to an open HDF5 file."""
        if len(crops) == 0:
            return

        n_new = len(crops)
        crop_ds = h5file["crops"]
        meta_ds = h5file["metadata"]

        n_existing = crop_ds.shape[0]
        crop_ds.resize(n_existing + n_new, axis=0)
        meta_ds.resize(n_existing + n_new, axis=0)

        crop_array = np.stack(crops, axis=0)
        meta_array = np.array(metadata_rows, dtype=METADATA_DTYPE)

        crop_ds[n_existing : n_existing + n_new] = crop_array
        meta_ds[n_existing : n_existing + n_new] = meta_array

        h5file.flush()


# ---------------------------------------------------------------------------
# Top-level experiment processor
# ---------------------------------------------------------------------------
def extract_experiment(
    image_dir: str | Path,
    output_dir: str | Path,
    model_path: str | Path = "/path/to/yolo11-obb.pt",
    batch_size: int = 16,
    crop_size: int = CROP_SIZE,
    conf_threshold: float = 0.25,
    focused_class_name: str = "Focused",
    device: str | None = None,
    compression: str | None = "gzip",
) -> Path:
    """Process all BMP frames in an experiment folder.

    Frames are sorted by their embedded timestamp. Only in-focus bacteria
    detections are kept.

    Parameters
    ----------
    image_dir : str or Path
        Folder containing ``*.bmp`` frames.
    output_dir : str or Path
        Folder where the HDF5 file will be written.
    model_path : str or Path
        YOLOv11-OBB weights file.
    batch_size : int
        Number of frames per YOLO inference batch.
    crop_size : int
        Side length of extracted crops.
    conf_threshold : float
        YOLO confidence threshold.
    focused_class_name : str
        YOLO class name for in-focus bacteria.
    device : str or None
        Inference device (``None`` = auto).
    compression : str or None
        HDF5 compression filter.

    Returns
    -------
    Path
        Path to the written HDF5 file.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    # Collect BMP frames and parse timestamps.
    frame_paths: list[Path] = []
    frame_timestamps: list[float] = []

    for p in sorted(image_dir.iterdir()):
        if p.suffix.lower() not in (".bmp", ".tiff", ".tif"):
            continue
        ts = parse_timestamp_from_filename(p.name)
        if ts is None:
            logger.warning("Could not parse timestamp from %s, skipping.", p.name)
            continue
        frame_paths.append(p)
        frame_timestamps.append(ts)

    if not frame_paths:
        raise FileNotFoundError(f"No valid image frames found in {image_dir}")

    # Sort by timestamp
    sort_idx = sorted(range(len(frame_timestamps)), key=lambda i: frame_timestamps[i])
    frame_paths = [frame_paths[i] for i in sort_idx]
    frame_timestamps = [frame_timestamps[i] for i in sort_idx]

    logger.info(
        "Found %d image frames in %s (%.1fs span)",
        len(frame_paths),
        image_dir,
        frame_timestamps[-1] - frame_timestamps[0] if len(frame_timestamps) > 1 else 0,
    )

    extractor = YOLOCropExtractor(
        model_path=model_path,
        crop_size=crop_size,
        batch_size=batch_size,
        conf_threshold=conf_threshold,
        focused_class_name=focused_class_name,
        device=device,
    )
    writer = HDF5CropWriter(
        crop_size=crop_size, compression=compression
    )

    # Use parent folder name as experiment name (the experiment folder)
    experiment_name = image_dir.parent.name
    h5_path = output_dir / f"{experiment_name}.h5"
    h5file = writer.create(h5_path)

    try:
        total_crops = 0
        for batch_start in tqdm(
            range(0, len(frame_paths), batch_size),
            desc=f"Processing {experiment_name}",
            unit="batch",
        ):
            batch_paths = frame_paths[batch_start : batch_start + batch_size]
            batch_timestamps = frame_timestamps[batch_start : batch_start + batch_size]
            frame_indices = list(range(batch_start, batch_start + len(batch_paths)))

            crops, metadata_rows = extractor.detect_and_crop(
                batch_paths, frame_indices, batch_timestamps
            )
            writer.append(h5file, crops, metadata_rows)
            total_crops += len(crops)

        logger.info(
            "Wrote %d focused crops to %s", total_crops, h5_path
        )
    finally:
        h5file.close()

    return h5_path
