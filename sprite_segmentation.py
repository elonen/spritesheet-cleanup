"""
Functions for segmenting sprites in a sprite sheet.
"""

from typing import List, Tuple
import cv2
import numpy as np

def segment_sprites(alpha: np.ndarray, min_size: int = 100, 
                   padding: int = 0) -> List[Tuple[int, int, int, int]]:
    """
    Segment individual sprites from a sprite sheet using the alpha channel.
    
    Args:
        alpha: Alpha channel of the image
        min_size: Minimum size (in pixels) for a region to be considered a sprite
        padding: Padding to add around each sprite bounding box
    
    Returns:
        List of sprite regions as (y1, y2, x1, x2) tuples
    """
    # Make binary mask from alpha channel
    _, binary = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Filter components by size and extract bounding boxes
    sprite_regions = []
    
    # Start from 1 to skip the background (label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area >= min_size:
            x = max(0, stats[i, cv2.CC_STAT_LEFT] - padding)
            y = max(0, stats[i, cv2.CC_STAT_TOP] - padding)
            w = stats[i, cv2.CC_STAT_WIDTH] + (padding * 2)
            h = stats[i, cv2.CC_STAT_HEIGHT] + (padding * 2)
            
            # Ensure we don't go beyond image boundaries
            if x + w > alpha.shape[1]:
                w = alpha.shape[1] - x
            if y + h > alpha.shape[0]:
                h = alpha.shape[0] - y
            
            sprite_regions.append((y, y + h, x, x + w))
    
    return sprite_regions

def expand_sprite_regions_to_grid(regions: List[Tuple[int, int, int, int]], 
                                 pixel_size: float) -> List[Tuple[int, int, int, int]]:
    """
    Expand sprite regions to align with the detected pixel grid.
    
    Args:
        regions: List of sprite regions as (y1, y2, x1, x2) tuples
        pixel_size: Detected pixel size
    
    Returns:
        List of adjusted sprite regions
    """
    expanded_regions = []
    
    for y1, y2, x1, x2 in regions:
        # Adjust bounds to align with pixel grid
        new_y1 = int(y1 // pixel_size * pixel_size)
        new_x1 = int(x1 // pixel_size * pixel_size)
        new_y2 = int(np.ceil(y2 / pixel_size) * pixel_size)
        new_x2 = int(np.ceil(x2 / pixel_size) * pixel_size)
        
        expanded_regions.append((new_y1, new_y2, new_x1, new_x2))
    
    return expanded_regions

def merge_overlapping_regions(regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """
    Merge overlapping sprite regions.
    
    Args:
        regions: List of sprite regions as (y1, y2, x1, x2) tuples
    
    Returns:
        List of merged sprite regions
    """
    if not regions:
        return []
    
    # Sort regions by y1, then x1
    sorted_regions = sorted(regions, key=lambda r: (r[0], r[2]))
    merged = [sorted_regions[0]]
    
    for current in sorted_regions[1:]:
        prev = merged[-1]
        
        # Check if current region overlaps with previous region
        if (current[0] <= prev[1] and current[2] <= prev[3] and
            current[1] >= prev[0] and current[3] >= prev[2]):
            # Merge the regions
            new_y1 = min(prev[0], current[0])
            new_y2 = max(prev[1], current[1])
            new_x1 = min(prev[2], current[2])
            new_x2 = max(prev[3], current[3])
            
            # Update the previous region with merged coordinates
            merged[-1] = (new_y1, new_y2, new_x1, new_x2)
        else:
            # No overlap, add current region to merged list
            merged.append(current)
    
    return merged
