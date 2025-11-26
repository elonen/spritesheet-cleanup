"""
Functions for restoring pixelated style in scaled-up pixel art.
"""
import numpy as np

def restore_smallscale_image(img: np.ndarray, h_lines: list[int], v_lines: list[int]) -> np.ndarray:
    """
    Restore the pixelated style of an image by finding the median color within
    detected grid cells.

    Args:
        img: Input image (BGRA)
        h_lines: Detected horizontal grid lines
        v_lines: Detected vertical grid lines

    Returns:
        Restored image
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

    # Process each grid cell
    for y in range(height):
        for x in range(width):
            y1, y2 = h_lines[y], h_lines[y + 1]
            x1, x2 = v_lines[x], v_lines[x + 1]

            # Skip empty cells
            if x2 <= x1 or y2 <= y1:
                continue

            # Extract the cell
            cell = img[y1:y2, x1:x2].copy()

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