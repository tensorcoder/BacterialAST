"""Microscopy-specific augmentations with DINO multi-crop strategy.

Implements custom augmentation transforms tailored for 96x96 grayscale
bacteria microscopy images, including a DINO-style multi-crop wrapper
that produces global and local crop views for self-supervised pre-training.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageFilter
from torch import Tensor
from torchvision import transforms as T
from torchvision.transforms import functional as TF


# ---------------------------------------------------------------------------
# Custom augmentation callables
# ---------------------------------------------------------------------------

class RandomIntensityJitter:
    """Random brightness and contrast jitter for single-channel grayscale.

    Brightness is applied as an additive shift; contrast is applied as a
    multiplicative factor around the image mean.  Both magnitudes are
    sampled uniformly from ``[-amount, +amount]``.

    Parameters
    ----------
    brightness : float
        Maximum absolute brightness shift, relative to [0, 1] range.
    contrast : float
        Maximum contrast factor deviation from 1.0.
    """

    def __init__(self, brightness: float = 0.3, contrast: float = 0.3) -> None:
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply random intensity jitter.

        Parameters
        ----------
        img : PIL.Image.Image
            Grayscale (mode ``"L"``) image.

        Returns
        -------
        PIL.Image.Image
        """
        # Work in float for precision.
        arr = np.asarray(img, dtype=np.float32) / 255.0

        # Brightness: additive.
        b = random.uniform(-self.brightness, self.brightness)
        arr = arr + b

        # Contrast: multiplicative around mean.
        c = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
        mean = arr.mean()
        arr = (arr - mean) * c + mean

        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(brightness={self.brightness}, "
            f"contrast={self.contrast})"
        )


class RandomGaussianNoise:
    """Additive Gaussian noise with random standard deviation.

    The noise std is sampled uniformly from ``std_range`` per call.  Pixel
    values are assumed in [0, 1] during noise addition and are clipped back.

    Parameters
    ----------
    std_range : tuple[float, float]
        ``(min_std, max_std)`` range for the noise standard deviation,
        expressed relative to a [0, 1] pixel range.
    """

    def __init__(self, std_range: tuple[float, float] = (0.0, 0.05)) -> None:
        self.std_range = std_range

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        std = random.uniform(*self.std_range)
        if std > 0:
            noise = np.random.normal(0.0, std, size=arr.shape).astype(np.float32)
            arr = arr + noise
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std_range={self.std_range})"


class RandomDefocusBlur:
    """Simulate microscope defocus with a disk or Gaussian blur.

    A random blur radius is sampled uniformly from ``radius_range``.  If the
    sampled radius is less than 0.5 the image is returned unchanged.  For
    radii up to 2 a Gaussian blur is used; for larger radii a disk (pillow
    ``BLUR``) kernel approximation is applied followed by Gaussian smoothing.

    Parameters
    ----------
    radius_range : tuple[int, int]
        ``(min_radius, max_radius)`` in pixels.
    """

    def __init__(self, radius_range: tuple[int, int] = (0, 3)) -> None:
        self.radius_range = radius_range

    @staticmethod
    def _disk_kernel(radius: int) -> ImageFilter.Kernel:
        """Create a circular averaging (disk) convolution kernel.

        Parameters
        ----------
        radius : int
            Kernel radius in pixels (kernel side = ``2 * radius + 1``).

        Returns
        -------
        ImageFilter.Kernel
        """
        size = 2 * radius + 1
        kernel_vals: list[int] = []
        for y in range(size):
            for x in range(size):
                dx = x - radius
                dy = y - radius
                if dx * dx + dy * dy <= radius * radius:
                    kernel_vals.append(1)
                else:
                    kernel_vals.append(0)
        total = sum(kernel_vals)
        if total == 0:
            total = 1
        return ImageFilter.Kernel(
            size=(size, size),
            kernel=kernel_vals,
            scale=total,
            offset=0,
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        radius = random.uniform(*self.radius_range)
        if radius < 0.5:
            return img

        if radius <= 2.0:
            # Simple Gaussian blur.
            return img.filter(ImageFilter.GaussianBlur(radius=radius))

        # Disk blur approximation for larger radii.
        int_radius = max(1, int(round(radius)))
        disk = self._disk_kernel(int_radius)
        blurred = img.filter(disk)
        # Light Gaussian smoothing to remove ringing from the hard disk edge.
        blurred = blurred.filter(ImageFilter.GaussianBlur(radius=0.5))
        return blurred

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(radius_range={self.radius_range})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_pil_grayscale(img: Image.Image | NDArray[np.uint8]) -> Image.Image:
    """Convert input to a PIL grayscale (``"L"``) image.

    Accepts:
    - PIL ``Image`` in any mode (will be converted to ``"L"``).
    - NumPy uint8 array of shape ``(H, W)`` or ``(H, W, 1)``.
    """
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        return Image.fromarray(img, mode="L")
    if isinstance(img, Image.Image):
        if img.mode != "L":
            return img.convert("L")
        return img
    raise TypeError(f"Unsupported image type: {type(img)}")


def _build_augmentation_pipeline(
    brightness: float = 0.3,
    contrast: float = 0.3,
    noise_std_range: tuple[float, float] = (0.0, 0.05),
    defocus_range: tuple[int, int] = (0, 3),
) -> list:
    """Build the list of microscopy augmentations (applied *before* crop)."""
    return [
        RandomIntensityJitter(brightness=brightness, contrast=contrast),
        RandomGaussianNoise(std_range=noise_std_range),
        RandomDefocusBlur(radius_range=defocus_range),
    ]


# ---------------------------------------------------------------------------
# DINO multi-crop augmentation
# ---------------------------------------------------------------------------

class DINOMicroscopyAugmentation:
    """DINO-style multi-crop augmentation for 96x96 grayscale microscopy.

    Produces:
    - 2 *global* crops at ``(96, 96)`` with scale ``(0.7, 1.0)``.
    - 6 *local* crops at ``(48, 48)`` with scale ``(0.3, 0.6)``.

    Every crop goes through:
    1. Random resized crop (at the specified scale and output size).
    2. Random rotation up to 180 degrees.
    3. Random horizontal and vertical flips.
    4. Microscopy-specific intensity / noise / blur augmentations.
    5. Conversion to float tensor and normalisation (mean=0.5, std=0.25).

    Parameters
    ----------
    global_crop_size : int
        Side length of global crops (default 96).
    local_crop_size : int
        Side length of local crops (default 48).
    global_scale : tuple[float, float]
        Scale range for global crops (default ``(0.7, 1.0)``).
    local_scale : tuple[float, float]
        Scale range for local crops (default ``(0.3, 0.6)``).
    n_global_crops : int
        Number of global crops (default 2).
    n_local_crops : int
        Number of local crops (default 6).
    brightness : float
        Brightness jitter magnitude.
    contrast : float
        Contrast jitter magnitude.
    noise_std_range : tuple[float, float]
        Gaussian noise standard deviation range.
    defocus_range : tuple[int, int]
        Defocus blur radius range.
    mean : Sequence[float]
        Normalisation mean (default ``[0.5]``).
    std : Sequence[float]
        Normalisation std (default ``[0.25]``).
    """

    def __init__(
        self,
        global_crop_size: int = 96,
        local_crop_size: int = 48,
        global_scale: tuple[float, float] = (0.7, 1.0),
        local_scale: tuple[float, float] = (0.3, 0.6),
        n_global_crops: int = 2,
        n_local_crops: int = 6,
        brightness: float = 0.3,
        contrast: float = 0.3,
        noise_std_range: tuple[float, float] = (0.0, 0.05),
        defocus_range: tuple[int, int] = (0, 3),
        mean: Sequence[float] = (0.5,),
        std: Sequence[float] = (0.25,),
    ) -> None:
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops

        microscopy_augs = _build_augmentation_pipeline(
            brightness=brightness,
            contrast=contrast,
            noise_std_range=noise_std_range,
            defocus_range=defocus_range,
        )

        # -- Global crop transform -------------------------------------------
        self.global_transform = T.Compose(
            [
                T.RandomResizedCrop(
                    size=global_crop_size,
                    scale=global_scale,
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.RandomRotation(degrees=180),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                *microscopy_augs,
                T.ToTensor(),  # -> (1, H, W) float32 in [0, 1]
                T.Normalize(mean=list(mean), std=list(std)),
            ]
        )

        # -- Local crop transform --------------------------------------------
        # Build a *separate* set of augmentation instances so that
        # global and local crops draw independent random parameters.
        microscopy_augs_local = _build_augmentation_pipeline(
            brightness=brightness,
            contrast=contrast,
            noise_std_range=noise_std_range,
            defocus_range=defocus_range,
        )

        self.local_transform = T.Compose(
            [
                T.RandomResizedCrop(
                    size=local_crop_size,
                    scale=local_scale,
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.RandomRotation(degrees=180),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                *microscopy_augs_local,
                T.ToTensor(),
                T.Normalize(mean=list(mean), std=list(std)),
            ]
        )

    def __call__(
        self, img: Image.Image | NDArray[np.uint8]
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Generate multi-crop views from a single input image.

        Parameters
        ----------
        img : PIL.Image.Image or ndarray
            Grayscale input image (expected 96x96 uint8, but any size is
            accepted -- ``RandomResizedCrop`` handles scaling).

        Returns
        -------
        global_crops : list[Tensor]
            ``n_global_crops`` tensors of shape ``(1, 96, 96)``.
        local_crops : list[Tensor]
            ``n_local_crops`` tensors of shape ``(1, 48, 48)``.
        """
        img = _ensure_pil_grayscale(img)

        global_crops: list[Tensor] = [
            self.global_transform(img) for _ in range(self.n_global_crops)
        ]
        local_crops: list[Tensor] = [
            self.local_transform(img) for _ in range(self.n_local_crops)
        ]
        return global_crops, local_crops

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_global={self.n_global_crops}, n_local={self.n_local_crops})"
        )
