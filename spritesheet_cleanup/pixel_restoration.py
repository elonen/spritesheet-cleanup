"""
Functions for restoring pixelated style in scaled-up pixel art.

A "cell" is a grid rectangle in the high-resolution input that represents
a single pixel in the restored output.
"""
import numpy as np

def restore_smallscale_image(
    img: np.ndarray,
    h_lines: list[int],
    v_lines: list[int],
    sample_center_pct: float = 60.0
) -> np.ndarray:
    """
    Restore the pixelated style of an image by finding the median color within
    detected grid cells.

    Args:
        img: Input image (BGRA)
        h_lines: Detected horizontal grid lines
        v_lines: Detected vertical grid lines
        sample_center_pct: Percentage of each cell (from center) to sample (0-100).
                          100% uses the full cell, 60% uses the center 60% width
                          and 60% height of each cell. This helps avoid color
                          bleeding from adjacent cells.

    Returns:
        Restored image with one output pixel per input cell
    """
    assert len(h_lines) > 1 and len(v_lines) > 1, "Grid lines must be detected"
    assert img.shape[2] == 4, "Image must have 4 channels (BGRA)"

    width = len(v_lines) - 1
    height = len(h_lines) - 1

    v_lines = sorted(v_lines)
    v_lines.append(img.shape[1])
    h_lines = sorted(h_lines)
    h_lines.append(img.shape[0])

    # Fix: Create array with height as first dimension, width as second
    restored = np.zeros((height, width, 4), dtype=np.uint8)

    # Clamp and convert sample_center_pct to a ratio
    sample_ratio = max(0.0, min(100.0, sample_center_pct)) / 100.0

    # Process each grid cell
    for y in range(height):
        for x in range(width):
            y1, y2 = h_lines[y], h_lines[y + 1]
            x1, x2 = v_lines[x], v_lines[x + 1]

            # Skip empty cells
            if x2 <= x1 or y2 <= y1:
                continue

            # Calculate center region to sample (to avoid color bleeding from edges)
            cell_w = x2 - x1
            cell_h = y2 - y1
            margin_x = round((cell_w * (1.0 - sample_ratio)) / 2.0)
            margin_y = round((cell_h * (1.0 - sample_ratio)) / 2.0)

            sample_x1 = x1 + margin_x
            sample_x2 = x2 - margin_x
            sample_y1 = y1 + margin_y
            sample_y2 = y2 - margin_y

            # Fall back to full cell if sampled region is too small
            if sample_x2 <= sample_x1:
                sample_x1, sample_x2 = x1, x2
            if sample_y2 <= sample_y1:
                sample_y1, sample_y2 = y1, y2

            # Extract the center region of the cell
            cell = img[sample_y1:sample_y2, sample_x1:sample_x2].copy()

            # Skip cells with no content
            if cell.size == 0:
                continue

            # Calculate dominant color using median
            b = np.median(cell[:,:,0])
            g = np.median(cell[:,:,1])
            r = np.median(cell[:,:,2])
            a = np.median(cell[:,:,3])

            # Decide alpha based on what percentage of the cell is opaque
            restored[y, x] = [b, g, r, a]

    return restored