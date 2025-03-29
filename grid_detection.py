"""
Functions for detecting pixel grid structure in scaled-up pixel art.
"""

from typing import List, Optional, Tuple, Dict
import cv2
import numpy as np
from collections import Counter

def detect_grid_parameters(img: np.ndarray, initial_estimate: Optional[float] = None) -> float:
    """
    Detect the original pixel grid size in a scaled-up pixel art image.
    
    Args:
        img: Input image (can be BGRA)
        initial_estimate: Initial estimate of the pixel size, if available
    
    Returns:
        Estimated pixel size (scale factor)
    """
    # Convert to grayscale if it has more than 1 channel
    if len(img.shape) > 2:
        # Use only RGB channels for edge detection, not alpha
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Enhance edges
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use Hough transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                          minLineLength=20, maxLineGap=10)
    
    if lines is None or len(lines) == 0:
        print("Warning: No lines detected, using initial estimate or default value")
        return initial_estimate if initial_estimate is not None else 8.0
    
    # Extract line coordinates and calculate slopes to identify horizontal and vertical lines
    horizontal_diffs: List[int] = []
    vertical_diffs: List[int] = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate slope
        if x2 - x1 == 0:  # Vertical line
            continue
            
        slope = (y2 - y1) / (x2 - x1)
        
        # Categorize as horizontal or vertical based on slope
        if abs(slope) < 0.1:  # Horizontal line
            horizontal_diffs.append(y1)
            horizontal_diffs.append(y2)
        elif abs(slope) > 10:  # Vertical line
            vertical_diffs.append(x1)
            vertical_diffs.append(x2)
    
    # Calculate the differences between adjacent coordinates
    pixel_sizes: List[int] = []
    
    if horizontal_diffs:
        horizontal_diffs.sort()
        for i in range(1, len(horizontal_diffs)):
            diff = horizontal_diffs[i] - horizontal_diffs[i-1]
            if diff > 1:  # Ignore differences that are too small
                pixel_sizes.append(diff)
    
    if vertical_diffs:
        vertical_diffs.sort()
        for i in range(1, len(vertical_diffs)):
            diff = vertical_diffs[i] - vertical_diffs[i-1]
            if diff > 1:  # Ignore differences that are too small
                pixel_sizes.append(diff)
    
    # Find the most common difference, which likely corresponds to the pixel size
    if not pixel_sizes:
        print("Warning: Could not determine pixel size, using initial estimate or default value")
        return initial_estimate if initial_estimate is not None else 8.0
    
    # Use a more sophisticated approach to find the most common pixel size
    # by grouping similar sizes together
    grouped_sizes: Dict[int, List[int]] = {}
    tolerance = 2  # Allow for some variation in detected sizes
    
    for size in pixel_sizes:
        matched = False
        for group_key in grouped_sizes:
            if abs(size - group_key) <= tolerance:
                grouped_sizes[group_key].append(size)
                matched = True
                break
        
        if not matched:
            grouped_sizes[size] = [size]
    
    # Find the group with the most elements
    most_common_group = max(grouped_sizes.items(), key=lambda x: len(x[1]))
    most_common_size = most_common_group[0]
    
    # If we have an initial estimate, use it to validate our detection
    if initial_estimate is not None:
        # If the detected size is very different from the initial estimate,
        # something might be wrong with the detection
        if abs(most_common_size - initial_estimate) > initial_estimate * 0.5:
            print(f"Warning: Detected size ({most_common_size}) differs significantly from initial estimate ({initial_estimate})")
            # Use a weighted average, favoring the initial estimate
            most_common_size = (initial_estimate * 0.7) + (most_common_size * 0.3)
    
    return float(most_common_size)

def refine_grid(img: np.ndarray, pixel_size: float) -> Tuple[List[int], List[int]]:
    """
    Refine the grid detection by finding the best alignment of horizontal and vertical lines.
    
    Args:
        img: Input image
        pixel_size: Estimated pixel size
    
    Returns:
        Tuple of (horizontal grid lines, vertical grid lines)
    """
    height, width = img.shape[:2]
    
    # Generate potential grid lines
    potential_h_lines = np.arange(0, height, pixel_size)
    potential_v_lines = np.arange(0, width, pixel_size)
    
    # Convert to grayscale if it has more than 1 channel
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Enhance edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Score each potential line by checking for edges along it
    h_scores = []
    for y in potential_h_lines:
        y = int(y)
        if 0 <= y < height:
            score = np.sum(edges[y, :])
            h_scores.append((y, score))
    
    v_scores = []
    for x in potential_v_lines:
        x = int(x)
        if 0 <= x < width:
            score = np.sum(edges[:, x])
            v_scores.append((x, score))
    
    # Sort by score and extract the top-scoring lines
    h_scores.sort(key=lambda x: x[1], reverse=True)
    v_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take the top 50% of lines
    h_lines = [y for y, _ in h_scores[:len(h_scores)//2]]
    v_lines = [x for x, _ in v_scores[:len(v_scores)//2]]
    
    # Sort lines by position
    h_lines.sort()
    v_lines.sort()
    
    return h_lines, v_lines
