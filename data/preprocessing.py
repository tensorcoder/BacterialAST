"""YOLO-OBB detection, oriented crop extraction, and HDF5 storage.

Processes raw BMP microscopy frames through YOLOv11-OBB to detect bacteria,
rectifies oriented bounding box crops via affine transform, resizes to 96x96
grayscale, and persists them in per-experiment HDF5 files with structured
metadata.
"""

from __future__ import annotations

import logging
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

CROP_SIZE: int = 96


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


# ---------------------------------------------------------------------------
# Structured NumPy dtype mirroring OBBDetection (used inside HDF5)
# ---------------------------------------------------------------------------
METADATA_DTYPE = np.dtype(
    [
        ("frame_idx", np.int32),
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
    device : str | None
        Device string forwarded to YOLO (e.g. ``"cuda:0"``, ``"cpu"``).
        *None* lets Ultralytics choose automatically.
    """

    def __init__(
        self,
        model_path: str | Path = "/path/to/yolo11-obb.pt",  # TODO: Set actual YOLO weights path
        crop_size: int = CROP_SIZE,
        batch_size: int = 16,
        conf_threshold: float = 0.25,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.device = device

        self._model: YOLO | None = None

    # -- lazy model loading ---------------------------------------------------

    @property
    def model(self) -> YOLO:
        """Lazily load the YOLO model on first access."""
        if self._model is None:
            logger.info("Loading YOLO-OBB model from %s", self.model_path)
            self._model = YOLO(str(self.model_path))
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
        """Extract an axis-aligned crop from an oriented bounding box.

        The region defined by (cx, cy, w, h, angle) is rectified using an
        affine warp so that the OBB becomes axis-aligned, then resized to
        ``(crop_size, crop_size)``.

        Parameters
        ----------
        image : ndarray
            Source image (grayscale uint8, H x W).
        cx, cy : float
            Centre coordinates of the OBB.
        w, h : float
            Width and height of the OBB (in pixels, *before* rotation).
        angle : float
            Counter-clockwise rotation angle in degrees.
        crop_size : int
            Output square side length.

        Returns
        -------
        ndarray
            Rectified crop of shape ``(crop_size, crop_size)``, uint8.
        """
        # Rotation matrix that *undoes* the OBB angle around its centre.
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

        # After rotation the OBB is axis-aligned -- extract the rectangle.
        half_w, half_h = w / 2.0, h / 2.0
        x1 = max(int(round(cx - half_w)), 0)
        y1 = max(int(round(cy - half_h)), 0)
        x2 = min(int(round(cx + half_w)), img_w)
        y2 = min(int(round(cy + half_h)), img_h)

        crop = rotated[y1:y2, x1:x2]

        # Guard against degenerate boxes that collapse to zero size.
        if crop.size == 0:
            return np.zeros((crop_size, crop_size), dtype=np.uint8)

        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        return crop

    def _parse_results(
        self,
        results: list,
        frame_indices: list[int],
    ) -> list[tuple[int, list[OBBDetection]]]:
        """Parse Ultralytics OBB result objects into ``OBBDetection`` lists.

        Parameters
        ----------
        results : list
            Ultralytics ``Results`` objects, one per frame.
        frame_indices : list[int]
            Corresponding frame indices in temporal order.

        Returns
        -------
        list of (frame_idx, detections) tuples.
        """
        parsed: list[tuple[int, list[OBBDetection]]] = []
        for frame_idx, result in zip(frame_indices, results):
            detections: list[OBBDetection] = []
            if result.obb is None:
                parsed.append((frame_idx, detections))
                continue

            obb_data = result.obb  # Ultralytics OBB object
            # obb_data.xywhr: (N, 5) tensor  [cx, cy, w, h, angle_rad]
            # obb_data.conf:  (N,) tensor
            boxes = obb_data.xywhr.cpu().numpy()  # (N, 5)
            confs = obb_data.conf.cpu().numpy()  # (N,)

            for det_id, (box, conf) in enumerate(zip(boxes, confs)):
                if conf < self.conf_threshold:
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
                    )
                )
            parsed.append((frame_idx, detections))
        return parsed

    def detect_and_crop(
        self,
        images: Sequence[NDArray[np.uint8]],
        frame_indices: list[int],
    ) -> tuple[list[NDArray[np.uint8]], list[np.void]]:
        """Run YOLO-OBB on a batch of images, returning crops and metadata.

        Parameters
        ----------
        images : sequence of ndarray
            Grayscale uint8 images (H x W each).
        frame_indices : list[int]
            Frame indices corresponding to each image.

        Returns
        -------
        crops : list of ndarray
            Each element is a ``(crop_size, crop_size)`` uint8 array.
        metadata_rows : list of np.void
            Structured array rows matching ``METADATA_DTYPE``.
        """
        results = self.model.predict(
            source=list(images),
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        parsed = self._parse_results(results, frame_indices)

        crops: list[NDArray[np.uint8]] = []
        metadata_rows: list[np.void] = []

        for frame_idx, detections in parsed:
            # Find the source image for this frame.
            src_idx = frame_indices.index(frame_idx)
            image = images[src_idx]
            # Ensure grayscale.
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            for det in detections:
                crop = self._rectify_obb_crop(
                    image, det.cx, det.cy, det.w, det.h, det.angle, self.crop_size
                )
                crops.append(crop)
                row = np.array(
                    (
                        frame_idx,
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
                metadata_rows.append(row[()])  # extract scalar np.void
        return crops, metadata_rows


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------
class HDF5CropWriter:
    """Write extracted crops and metadata to an HDF5 file.

    The file layout is::

        /crops      (N, 96, 96)  uint8       -- resizable along axis 0
        /metadata   (N,)         structured   -- ``METADATA_DTYPE``

    Parameters
    ----------
    crop_size : int
        Side length of each square crop (must match extractor output).
    chunk_size : int
        HDF5 chunk size along axis 0 for incremental writes.
    compression : str | None
        HDF5 compression filter (e.g. ``"gzip"``).  *None* disables
        compression for maximum write speed.
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
        """Create (or overwrite) an HDF5 file with the expected datasets.

        Parameters
        ----------
        path : str or Path
            Destination HDF5 file path.

        Returns
        -------
        h5py.File
            The opened (writable) HDF5 file handle.  Caller is responsible
            for closing the file (or using a context manager externally).
        """
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
        """Append a batch of crops and metadata to an open HDF5 file.

        Parameters
        ----------
        h5file : h5py.File
            Writable HDF5 file (as returned by :meth:`create`).
        crops : list of ndarray
            ``(crop_size, crop_size)`` uint8 arrays.
        metadata_rows : list of np.void
            Structured scalars matching ``METADATA_DTYPE``.
        """
        if len(crops) == 0:
            return

        n_new = len(crops)
        crop_ds = h5file["crops"]
        meta_ds = h5file["metadata"]

        n_existing = crop_ds.shape[0]
        crop_ds.resize(n_existing + n_new, axis=0)
        meta_ds.resize(n_existing + n_new, axis=0)

        crop_array = np.stack(crops, axis=0)  # (n_new, crop_size, crop_size)
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
    model_path: str | Path = "/path/to/yolo11-obb.pt",  # TODO: Set actual YOLO weights path
    batch_size: int = 16,
    crop_size: int = CROP_SIZE,
    conf_threshold: float = 0.25,
    device: str | None = None,
    compression: str | None = "gzip",
) -> Path:
    """Process all BMP frames in an experiment folder.

    Frames are sorted lexicographically by filename (expected to contain
    datetime stamps so that lexicographic order equals temporal order).

    Parameters
    ----------
    image_dir : str or Path
        Folder containing ``*.bmp`` frames.
    output_dir : str or Path
        Folder where the HDF5 file will be written.  The file is named
        after the experiment folder: ``<experiment_name>.h5``.
    model_path : str or Path
        YOLOv11-OBB weights file.
    batch_size : int
        Number of frames per YOLO inference batch.
    crop_size : int
        Side length of extracted crops.
    conf_threshold : float
        YOLO confidence threshold.
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

    # Collect and sort BMP frames (case-insensitive extension match).
    frame_paths: list[Path] = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() == ".bmp"
    )
    if not frame_paths:
        raise FileNotFoundError(f"No BMP frames found in {image_dir}")

    logger.info(
        "Found %d BMP frames in %s", len(frame_paths), image_dir
    )

    extractor = YOLOCropExtractor(
        model_path=model_path,
        crop_size=crop_size,
        batch_size=batch_size,
        conf_threshold=conf_threshold,
        device=device,
    )
    writer = HDF5CropWriter(
        crop_size=crop_size, compression=compression
    )

    experiment_name = image_dir.name
    h5_path = output_dir / f"{experiment_name}.h5"
    h5file = writer.create(h5_path)

    try:
        total_crops = 0
        # Process in batches.
        for batch_start in tqdm(
            range(0, len(frame_paths), batch_size),
            desc=f"Processing {experiment_name}",
            unit="batch",
        ):
            batch_paths = frame_paths[batch_start : batch_start + batch_size]
            frame_indices = list(range(batch_start, batch_start + len(batch_paths)))

            images: list[NDArray[np.uint8]] = []
            for p in batch_paths:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.warning("Failed to read %s, skipping.", p)
                    continue
                images.append(img)

            if not images:
                continue

            # Trim frame_indices to match successfully loaded images.
            frame_indices = frame_indices[: len(images)]

            crops, metadata_rows = extractor.detect_and_crop(images, frame_indices)
            writer.append(h5file, crops, metadata_rows)
            total_crops += len(crops)

        logger.info(
            "Wrote %d crops to %s", total_crops, h5_path
        )
    finally:
        h5file.close()

    return h5_path
