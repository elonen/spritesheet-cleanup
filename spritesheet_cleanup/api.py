#!/usr/bin/env python3
"""
Public API for the Pixel Art Restoration Library.

This module provides the main interface for programmatic use of the sprite
extraction and restoration functionality.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

from spritesheet_cleanup.alpha_processing import clean_alpha_channel
from spritesheet_cleanup.grid_detection import estimate_grid_size, refine_grid
from spritesheet_cleanup.sprite_segmentation import segment_sprites
from spritesheet_cleanup.pixel_restoration import restore_smallscale_image
from spritesheet_cleanup.grid_visualization import visualize_grid


@dataclass
class ProcessedImage:
    """
    A processed image with metadata from the sprite extraction pipeline.

    Attributes:
        image: The image data as a numpy array (BGRA format, uint8)
        name: Descriptive name for the image (e.g., "sprite_0", "debug_cleaned_alpha")
        bbox: Original bounding box for sprites as (y1, y2, x1, x2), or None for debug images
        is_debug: True if this is a debug/intermediate image, False for final output sprites
        metadata: Additional metadata (e.g., pixel size estimates, grid info)
    """
    image: np.ndarray
    name: str
    bbox: tuple[int, int, int, int] | None
    is_debug: bool
    metadata: dict[str, float | int] | None = None


# Backwards compatibility alias
@dataclass
class SpriteResult:
    """
    Backwards compatibility class for sprite extraction results.

    Deprecated: Use process_spritesheet() generator API instead.

    Attributes:
        image: Restored sprite as a BGRA numpy array (uint8)
        bbox: Original bounding box in source image as (y1, y2, x1, x2)
              where y1, y2 are row indices and x1, x2 are column indices
    """
    image: np.ndarray
    bbox: tuple[int, int, int, int]


def process_spritesheet(
    image: np.ndarray | None,
    *,
    min_sprite_size: float = 2.0,
    pixel_w_guess: float | None = None,
    pixel_h_guess: float | None = None,
    pixel_w_slop: float = 0.33,
    pixel_h_slop: float = 0.33,
    no_segment: bool = False,
    bilateral_filter: bool = False,
    debug: bool = False
) -> Generator[ProcessedImage, None, None]:
    """
    Process a spritesheet image and yield sprites and debug images as they're produced.

    Takes a scaled-up, possibly noisy pixel art spritesheet and restores it to
    its original (lower) resolution. If the image has an alpha channel, it will
    by default be segmented into individual sprites, with grid refinement per sprite.

    This function is a generator that yields ProcessedImage objects as they're
    created, allowing the caller to handle each image (save, display, etc.) as desired.

    Args:
        image: Input image as numpy array in BGR or BGRA format (uint8).
               Must be 3D array with shape (height, width, 3) or (height, width, 4).
        min_sprite_size: Minimum size of sprite in pixels after restoration.
                        Sprites smaller than this will be filtered out.
        pixel_w_guess: Optional initial guess for pixel width in the source image.
                      If not provided, it will be estimated automatically.
        pixel_h_guess: Optional initial guess for pixel height in the source image.
                      If not provided, it will be estimated automatically.
        pixel_w_slop: Multiplier for pixel width guess tolerance. Higher values
                     allow more variation in detected pixel widths.
        pixel_h_slop: Multiplier for pixel height guess tolerance. Higher values
                     allow more variation in detected pixel heights.
        no_segment: If True, skip sprite segmentation and process the entire image
                   as a single sprite.
        bilateral_filter: If True, apply bilateral noise filtering before processing.
                         Useful for images with JPEG artifacts or noise.
        debug: If True, yield intermediate processing images for debugging.

    Yields:
        ProcessedImage objects, each containing:
        - image: Image data as BGRA numpy array (uint8)
        - name: Descriptive name (e.g., "sprite_0", "debug_cleaned_alpha")
        - bbox: Original bounding box as (y1, y2, x1, x2) for sprites, None for debug images
        - is_debug: True for debug images, False for final sprites
        - metadata: Optional dict with additional info (e.g., pixel size estimates)

        Images are yielded in processing order. Debug images (if enabled) are yielded
        as they're created, followed by final sprites in their source order.

    Raises:
        ValueError: If image is None or has invalid shape/dtype.

    Example:
        >>> import cv2
        >>> from spritesheet_cleanup import process_spritesheet
        >>>
        >>> # Load an image
        >>> img = cv2.imread("spritesheet.png", cv2.IMREAD_UNCHANGED)
        >>>
        >>> # Process it and handle images as they're produced
        >>> for result in process_spritesheet(img, debug=True):
        >>>     if result.is_debug:
        >>>         print(f"Debug: {result.name}")
        >>>         cv2.imshow(result.name, result.image)
        >>>     else:
        >>>         print(f"Sprite: {result.name} from {result.bbox}")
        >>>         cv2.imwrite(f"{result.name}.png", result.image)
    """
    # Validate input
    if image is None:
        raise ValueError("image cannot be None")

    if not isinstance(image, np.ndarray):
        raise ValueError(f"image must be a numpy array, got {type(image)}")

    if image.ndim != 3:
        raise ValueError(f"image must be 3D array (height, width, channels), got shape {image.shape}")

    if image.shape[2] not in (3, 4):
        raise ValueError(f"image must have 3 (BGR) or 4 (BGRA) channels, got {image.shape[2]}")

    if image.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {image.dtype}")

    # Make a copy to avoid modifying the input
    img = image.copy()

    # Apply bilateral filter to each channel to reduce noise
    if bilateral_filter:
        for i in range(img.shape[2]):
            img[:, :, i] = cv2.bilateralFilter(img[:, :, i], d=5, sigmaColor=75, sigmaSpace=75)

    # Check if the image has an alpha channel, add one if it doesn't
    if img.shape[2] == 3:
        alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    else:
        alpha = img[:, :, 3].copy()

    # Clean the alpha channel to remove noise
    cleaned_alpha = clean_alpha_channel(alpha)
    img[:, :, 3] = cleaned_alpha

    # Yield debug images if requested
    if debug:
        yield ProcessedImage(
            image=cleaned_alpha,
            name="debug_cleaned_alpha",
            bbox=None,
            is_debug=True,
            metadata=None
        )
        yield ProcessedImage(
            image=img.copy(),
            name="debug_image_with_cleaned_alpha",
            bbox=None,
            is_debug=True,
            metadata=None
        )

    # Estimate pixel size for the entire image
    pix_w, pix_h, w_std, h_std = estimate_grid_size(
        img,
        debug=debug,
        w_guess=pixel_w_guess,
        h_guess=pixel_h_guess,
        w_slop_mult=pixel_w_slop,
        h_slop_mult=pixel_h_slop
    )

    # Segment sprites using the minimum size derived from estimated pixel size
    if no_segment:
        sprite_regions = [(0, img.shape[0], 0, img.shape[1])]
    else:
        min_size = int((pix_w * pix_h) * min_sprite_size)
        sprite_regions = segment_sprites(cleaned_alpha, min_size)

        if debug:
            segment_sprites_img = img.copy()
            for region in sprite_regions:
                y1, y2, x1, x2 = region
                cv2.rectangle(segment_sprites_img, (x1, y1), (x2, y2), (0, 255, 0, 255), 1)
            yield ProcessedImage(
                image=segment_sprites_img,
                name="debug_segmented",
                bbox=None,
                is_debug=True,
                metadata={"num_sprites": len(sprite_regions)}
            )

    # Process each sprite, restore it to original pixel size
    for i, region in enumerate(sprite_regions):
        y1, y2, x1, x2 = region
        sprite = img[y1:y2, x1:x2].copy()

        h_lines, v_lines, _edges_img = refine_grid(sprite, pix_w, pix_h, w_std, h_std)

        # Yield debug images if requested
        if debug:
            yield ProcessedImage(
                image=sprite.copy(),
                name=f"debug_sprite_{i}_original",
                bbox=(int(y1), int(y2), int(x1), int(x2)),
                is_debug=True,
                metadata={"sprite_index": i}
            )

            grid_vis = visualize_grid(sprite, pix_w, pix_h, h_lines, v_lines)
            yield ProcessedImage(
                image=grid_vis,
                name=f"debug_sprite_{i}_grid_visualization",
                bbox=(int(y1), int(y2), int(x1), int(x2)),
                is_debug=True,
                metadata={"sprite_index": i}
            )

        restored_sprite = restore_smallscale_image(sprite, h_lines, v_lines)

        # Yield debug visualization of the restored sprite (upscaled)
        if debug:
            mean_pix_w = int(np.mean(np.diff(v_lines)) + 0.5)
            mean_pix_h = int(np.mean(np.diff(h_lines)) + 0.5)
            upscaled_sprite = cv2.resize(
                restored_sprite,
                (mean_pix_w * len(v_lines), mean_pix_h * len(h_lines)),
                interpolation=cv2.INTER_NEAREST
            )
            yield ProcessedImage(
                image=upscaled_sprite,
                name=f"debug_sprite_{i}_restored_upscaled",
                bbox=(int(y1), int(y2), int(x1), int(x2)),
                is_debug=True,
                metadata={"sprite_index": i, "upscaled": True}
            )

        # Yield the final restored sprite
        yield ProcessedImage(
            image=restored_sprite,
            name=f"sprite_{i}",
            bbox=(int(y1), int(y2), int(x1), int(x2)),
            is_debug=False,
            metadata={
                "sprite_index": i,
                "pixel_width": float(pix_w),
                "pixel_height": float(pix_h)
            }
        )
