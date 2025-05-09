"""
Functions for processing and cleaning alpha channels in images.
"""

import cv2
import numpy as np

def clean_alpha_channel(alpha: np.ndarray, threshold: int = 128,
                        kernel_size: int = 3) -> np.ndarray:
    """
    Clean up the alpha channel by applying thresholding and morphological operations.

    Args:
        alpha: The alpha channel to clean
        threshold: Threshold value for binary thresholding (0-255)
        kernel_size: Size of the kernel for morphological operations

    Returns:
        Cleaned alpha channel
    """
    # Apply binary thresholding to make alpha more definitive
    _, binary_alpha = cv2.threshold(alpha, threshold, 255, cv2.THRESH_BINARY)

    # Create kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Remove small noise with opening operation (erosion followed by dilation)
    cleaned_alpha = cv2.morphologyEx(binary_alpha, cv2.MORPH_OPEN, kernel)

    # Fill small holes with closing operation (dilation followed by erosion)
    cleaned_alpha = cv2.morphologyEx(cleaned_alpha, cv2.MORPH_CLOSE, kernel)

    return cleaned_alpha

def extract_foreground(img: np.ndarray, alpha_threshold: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the foreground pixels using the alpha channel.

    Args:
        img: BGRA image
        alpha_threshold: Threshold for determining foreground pixels

    Returns:
        Tuple of (foreground mask, foreground pixels)
    """
    # Extract alpha channel
    alpha = img[:, :, 3]

    # Create foreground mask
    foreground_mask = alpha > alpha_threshold

    # Extract foreground pixels
    foreground = img.copy()
    foreground[~foreground_mask] = 0

    return foreground_mask, foreground
