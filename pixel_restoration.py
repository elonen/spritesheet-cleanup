"""
Functions for restoring pixelated style in scaled-up pixel art.
"""

from typing import List, Tuple
import cv2
import numpy as np
import math
from grid_detection import refine_grid

def restore_pixelated_style(img: np.ndarray, pixel_size: float) -> np.ndarray:
    """
    Restore the pixelated style of an image by finding the dominant color within
    detected grid cells.

    Args:
        img: Input image (BGRA)
        pixel_size: Detected pixel size

    Returns:
        Restored pixelated image
    """
    # Create a more robust grid based on pixel size
    # Rather than trying to detect the grid lines perfectly, use a regular grid
    # that's well-aligned with the image dimensions
    height, width = img.shape[:2]

    # Calculate the number of grid cells that would fit
    num_rows = max(1, round(height / pixel_size))
    num_cols = max(1, round(width / pixel_size))

    # Recalculate the actual pixel size to ensure even grid
    actual_pixel_height = height / num_rows
    actual_pixel_width = width / num_cols

    # Create grid lines
    h_lines = [int(i * actual_pixel_height) for i in range(num_rows + 1)]
    v_lines = [int(i * actual_pixel_width) for i in range(num_cols + 1)]

    # Make sure the last line captures the image boundary
    h_lines[-1] = height
    v_lines[-1] = width

    # Create output image with same shape as input
    restored = np.zeros_like(img)

    # Process each grid cell
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            y1, y2 = h_lines[i], h_lines[i + 1]
            x1, x2 = v_lines[j], v_lines[j + 1]

            # Skip cells that are too small
            if y2 - y1 < 2 or x2 - x1 < 2:
                continue

            # Extract the cell
            cell = img[y1:y2, x1:x2].copy()

            # Get only opaque or mostly opaque pixels for color calculation
            opaque_mask = cell[:, :, 3] > 200

            # If the cell has enough opaque pixels, use them for color calculation
            if np.sum(opaque_mask) > (cell.shape[0] * cell.shape[1]) * 0.3:
                # Get dominant color using only fully opaque pixels
                opaque_pixels = cell[opaque_mask]

                if len(opaque_pixels) > 0:
                    # Take the median of the opaque pixels for each channel
                    # Median is more robust to outliers than mean
                    median_b = np.median(opaque_pixels[:, 0])
                    median_g = np.median(opaque_pixels[:, 1])
                    median_r = np.median(opaque_pixels[:, 2])

                    # Set the color in the restored image
                    restored[y1:y2, x1:x2, 0] = median_b
                    restored[y1:y2, x1:x2, 1] = median_g
                    restored[y1:y2, x1:x2, 2] = median_r
                    restored[y1:y2, x1:x2, 3] = 255  # Fully opaque
            else:
                # Check if the cell has any semi-transparent pixels
                semi_transparent_mask = (cell[:, :, 3] > 50) & (cell[:, :, 3] <= 200)

                if np.sum(semi_transparent_mask) > 0:
                    # If we have semi-transparent pixels but not enough opaque ones,
                    # use the semi-transparent pixels
                    semi_pixels = cell[semi_transparent_mask]

                    if len(semi_pixels) > 0:
                        # Use median of semi-transparent pixels
                        median_b = np.median(semi_pixels[:, 0])
                        median_g = np.median(semi_pixels[:, 1])
                        median_r = np.median(semi_pixels[:, 2])

                        # Average alpha of semi-transparent pixels
                        median_a = np.median(semi_pixels[:, 3])

                        # Set the color
                        restored[y1:y2, x1:x2, 0] = median_b
                        restored[y1:y2, x1:x2, 1] = median_g
                        restored[y1:y2, x1:x2, 2] = median_r
                        # Make it either fully transparent or fully opaque
                        restored[y1:y2, x1:x2, 3] = 255 if median_a > 128 else 0
                else:
                    # Cell is mostly transparent
                    restored[y1:y2, x1:x2, 3] = 0

    return restored

def quantize_colors(img: np.ndarray, num_colors: int = 32) -> np.ndarray:
    """
    Reduce the number of colors in an image using k-means clustering.

    Args:
        img: Input image (BGRA)
        num_colors: Number of colors to reduce to

    Returns:
        Image with reduced color palette
    """
    # Reshape the image to a list of pixels
    pixels = img[:, :, :3].reshape(-1, 3).astype(np.float32)

    # Only include non-transparent pixels
    alpha = img[:, :, 3].reshape(-1)
    valid_pixels = pixels[alpha > 128]

    if len(valid_pixels) == 0:
        return img.copy()

    # Define criteria and apply k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(valid_pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Create output image with the same shape as input
    quantized = img.copy()

    # Apply quantization only to non-transparent pixels
    mask = alpha > 128
    reshaped_mask = mask.reshape(-1)

    # Map each pixel to its nearest center
    quantized_pixels = np.zeros_like(pixels)
    quantized_pixels[reshaped_mask] = centers[labels.flatten()]

    # Reshape back to the original image shape
    quantized[:, :, :3] = quantized_pixels.reshape(img.shape[0], img.shape[1], 3)

    return quantized

def apply_median_filter_preserving_edges(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Apply a median filter to reduce noise while preserving edges.

    Args:
        img: Input image (BGRA)
        ksize: Kernel size for the median filter

    Returns:
        Filtered image
    """
    # Split channels
    b, g, r, a = cv2.split(img)

    # Apply median filter to color channels
    b_filtered = cv2.medianBlur(b, ksize)
    g_filtered = cv2.medianBlur(g, ksize)
    r_filtered = cv2.medianBlur(r, ksize)

    # Don't filter alpha channel
    a_filtered = a

    # Merge channels back
    filtered = cv2.merge([b_filtered, g_filtered, r_filtered, a_filtered])

    return filtered
