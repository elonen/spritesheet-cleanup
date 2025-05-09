"""
Functions for segmenting sprites in a sprite sheet.
"""

import cv2
import numpy as np

def segment_sprites(alpha: np.ndarray, min_size: int) -> list[tuple[int, int, int, int]]:
    """
    Segment individual sprites from a sprite sheet using the alpha channel.

    Args:
        alpha: Alpha channel of the image
        min_size: Minimum size (in pixels) for a region to be considered a sprite

    Returns:
        List of sprite regions as (y1, y2, x1, x2) tuples
    """
    # Make binary mask from alpha channel
    _, binary = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Filter components by size and extract bounding boxes
    sprite_regions = []

    # Start from 1 to skip the background (label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_size:
            x = max(0, stats[i, cv2.CC_STAT_LEFT])
            y = max(0, stats[i, cv2.CC_STAT_TOP])
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Ensure we don't go beyond image boundaries
            if x + w > alpha.shape[1]:
                w = alpha.shape[1] - x
            if y + h > alpha.shape[0]:
                h = alpha.shape[0] - y

            sprite_regions.append((y, y + h, x, x + w))

    return sprite_regions
