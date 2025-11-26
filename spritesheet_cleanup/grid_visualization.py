"""
Functions for visualizing the detected grid on images.
"""

import cv2
import numpy as np

def visualize_grid(img: np.ndarray, est_pixel_w: float, est_pixel_h: float, h_lines: list[int], v_lines: list[int],
                  output_path: str|None = None) -> np.ndarray:
    """
    Visualize the detected grid by overlaying magenta lines on the image.

    Args:
        img: Input image (BGRA)
        h_lines: Detected horizontal grid lines
        v_lines: Detected vertical grid lines
        output_path: Path to save the visualization (optional)

    Returns:
        Image with grid overlay
    """
    # Create a copy of the image for visualization
    vis_img = img.copy()

    # Convert to BGR if it has 4 channels (BGRA)
    if vis_img.shape[2] == 4:
        # Create a white background
        bg = np.ones((vis_img.shape[0], vis_img.shape[1], 3), dtype=np.uint8) * 255
        # Compute alpha blending
        alpha = vis_img[:, :, 3:4].astype(float) / 255
        vis_img = (vis_img[:, :, :3] * alpha + bg * (1 - alpha)).astype(np.uint8)

    height, width = vis_img.shape[:2]

    # Add a second visualization with a uniform grid for comparison
    uniform_img = vis_img.copy()

    # Calculate uniform grid lines
    num_rows = max(1, round(height / est_pixel_h))
    num_cols = max(1, round(width / est_pixel_w))

    uniform_h_lines = [int(i * height / num_rows) for i in range(num_rows + 1)]
    uniform_v_lines = [int(i * width / num_cols) for i in range(num_cols + 1)]

    for y in h_lines:
        cv2.line(vis_img, (0, y), (width, y), (255, 0, 255), 1)  # Magenta

    for y in uniform_h_lines:
        cv2.line(uniform_img, (0, y), (width, y), (255, 255, 0), 1)  # Cyan

    for x in v_lines:
        cv2.line(vis_img, (x, 0), (x, height), (255, 0, 255), 1)  # Magenta

    for x in uniform_v_lines:
        cv2.line(uniform_img, (x, 0), (x, height), (255, 255, 0), 1)  # Cyan

    # Combine the two images side by side
    combined = np.hstack((vis_img, uniform_img))

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Detected Grid", (10, 30), font, 1, (255, 0, 255), 2)
    cv2.putText(combined, "Uniform Grid", (width + 10, 30), font, 1, (255, 255, 0), 2)

    if output_path:
        cv2.imwrite(output_path, combined)

    return combined
