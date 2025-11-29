import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit    # type: ignore


def estimate_grid_size(
        img: np.ndarray,
        debug: bool,
        w_guess: float|None = None,
        h_guess: float|None = None,
        w_slop_mult: float = 0.3,
        h_slop_mult: float = 0.3,
        ) -> tuple[float, float, float, float, ]:
    """
    Detect the cell grid size in a scaled-up pixel art image.

    Args:
        img: Input image (can be BGRA)
        initial_estimate: Initial estimate of the cell size, if available

    Returns:
        Estimated cell size and standard deviations for width and height
    """

    # Find edges using Sobel filter over all channels, then threshold to binary
    edges = _multi_channel_sobel(img)
    _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Denoise the binary edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    min_w, max_w = 0.0, img.shape[1]
    min_h, max_h = 0.0, img.shape[0]
    if w_guess is not None:
        min_w = max(0.0, w_guess - w_slop_mult * w_guess)
        max_w = min(edges.shape[1], w_guess + w_slop_mult * w_guess)
    if h_guess is not None:
        min_h = max(0.0, h_guess - h_slop_mult * h_guess)
        max_h = min(edges.shape[0], h_guess + h_slop_mult * h_guess)

    # Process rows and columns using a loop over dimensions
    results = []
    for direction in (0, 1):        # hor, ver
        distances: list[float] = []
        for i in range(edges.shape[direction]):
            # Get row or column based on direction
            line = edges[i, :] if direction == 0 else edges[:, i]

            # Find rising edges (0->1 transitions)
            rising_edges = np.where(np.diff(line.astype(int)) > 0)[0] + 1

            if len(rising_edges) > 1:
                # Get distances between consecutive rising edges
                diff_distances = np.diff(rising_edges)

                # Filter out distances that are too small or too large
                distances.extend([d for d in diff_distances if min_w <= d <= max_w] if direction == 1 else \
                                 [d for d in diff_distances if min_h <= d <= max_h])

        if len(distances) == 0:
            raise ValueError("Grid detection failed: No distances found.")

        # Create high-resolution histogram with 1.0 pixel bins
        bin_width = 1.0
        hist, bins = np.histogram(distances, bins=np.arange(0, max(distances)+bin_width, bin_width))

        # Visualize the histogram if debug is enabled
        if debug:
            plt.figure(figsize=(10, 6))
            plt.bar(bins[:-1], hist, width=np.diff(bins), align='edge', alpha=0.7)
            plt.title(f"{'Horizontal' if direction == 0 else 'Vertical'} Distances Histogram")
            plt.xlabel("Distance (pixels)")
            plt.ylabel("Frequency")
            plt.xlim(0, min(50, max(distances)))
            plt.grid(alpha=0.3)
            plt.show()

        # Find the peak bin
        peak_idx = np.argmax(hist)

        # Simple estimate from the peak bin
        simple_estimate = float((bins[peak_idx] + bins[peak_idx+1]) / 2)

        def gaussian(x, mean, sigma, amplitude):
            return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2))

        # Define fitting range - peak bin plus/minus 3 bins (or less if near edges)
        min_idx = max(0, int(peak_idx) - 3)
        max_idx = min(len(hist) - 1, int(peak_idx) + 3)

        # Get bin centers for the fitting range
        x_data = [(bins[i] + bins[i+1]) / 2 for i in range(min_idx, max_idx + 1)]
        y_data = hist[min_idx:max_idx + 1]

        # Only attempt fit if we have enough data points
        if len(x_data) >= 3:
            # Initial guess for parameters
            p0 = [simple_estimate, 1.0, hist[peak_idx]]

            # Fit the Gaussian
            popt, _ = curve_fit(gaussian, x_data, y_data, p0=p0)

            # Extract the parameters
            mean = popt[0]
            std_dev = abs(popt[1])  # Ensure positive

            # Visualize the fit if debug is enabled
            if debug:
                plt.figure(figsize=(10, 6))
                plt.bar(bins[:-1], hist, width=np.diff(bins), align='edge', alpha=0.7,
                        label='Histogram')

                # Plot the fitted Gaussian over a finer grid
                x_fine = np.linspace(bins[min_idx], bins[max_idx+1], 100)
                plt.plot(x_fine, gaussian(x_fine, *popt), 'r-', linewidth=2,
                            label=f'Gaussian fit (μ={mean:.2f}, σ={std_dev:.2f})')

                plt.axvline(x=mean, color='g', linestyle='--',
                            label=f'Mean = {mean:.2f}')

                plt.title(f"{'Horizontal' if direction == 0 else 'Vertical'} Gaussian Fit")
                plt.xlabel("Distance (pixels)")
                plt.ylabel("Frequency")
                plt.legend()
                plt.xlim(bins[min_idx] - 2, bins[max_idx+1] + 2)
                plt.grid(alpha=0.3)
                plt.show()

            # Store mean and standard deviation
            results.extend([mean, std_dev])
        else:
            # Not enough data points for fitting, use simple estimate
            print(f"Not enough data points for fitting gaussian in {'horizontal' if direction == 0 else 'vertical'} direction. Using simple estimate.")
            results.extend([simple_estimate, 1.0])

    return results[0], results[2], results[1], results[3]   # cell width, cell height, std_dev_w, std_dev_h


def refine_grid(img: np.ndarray, pix_w: float, pix_h: float, _std_dev_w: float, _std_dev_h: float) -> tuple[list[int], list[int], np.ndarray]:
    """
    Refine the grid detection by finding the best alignment of horizontal and vertical lines.

    Args:
        img: Input image
        pix_w: Estimated cell width
        pix_h: Estimated cell height
        std_dev_w: Standard deviation of width
        std_dev_h: Standard deviation of height

    Returns:
        Tuple of (horizontal grid lines, vertical grid lines)
    """
    edges = _multi_channel_sobel(img)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))   # Denoise

    results: tuple[list[int], list[int]] = ([], [])  # Horizontal and vertical results

    for dir in (0,1):   # Horizontal, Vertical
        if dir == 0:
            edge_img = edges
            avg_step = pix_h    # Gap between two horizontal lines = cell height
        else:
            edge_img = cv2.transpose(edges)
            avg_step = pix_w

        # std_dev = direction['std_dev']
        search_range_basic = int(avg_step * 0.6 / 2.0)
        empty_run_len = 0

        pos = 0
        while pos < edge_img.shape[0]:  # Loop through the image dimension
            results[dir].append(pos)

            expected_pos = pos + int(avg_step)
            best_score = 0.0
            best_pos = expected_pos

            empty_steps_taken = float(empty_run_len) / avg_step
            search_range = min(avg_step/2.0, search_range_basic * (1 + empty_steps_taken/6.0))

            for p_cand in range(expected_pos - int(search_range+0.5), expected_pos + int(search_range+0.5) + 1):
                if p_cand >= edge_img.shape[0] or p_cand < 0:
                    continue

                # Apply Gaussian weighting based on distance from expected position
                # distance = abs(p_cand - expected_pos)
                # gaussian_weight = np.exp(-(distance**2) / (2 * std_dev**2))

                # Calculate edge strength and score
                edge_score = np.sum(edge_img[p_cand])
                score = edge_score  # * gaussian_weight (if using)
                if score > best_score:
                    best_score = float(score)
                    best_pos = p_cand

                # Track how many empty pixels we have seen in a row
                if score < 0.001 * len(edge_img[p_cand]):
                    empty_run_len += 1
                else:
                    empty_run_len = 0

            pos = best_pos

    # Return horizontal and vertical gridlines
    return results[0], results[1], edges


def _multi_channel_sobel(image: np.ndarray) -> np.ndarray:
    """
    Apply Sobel filter to each channel of the image - including alpha - and combine the results.
    This highlights edges while reducing noise.
    """
    channels = cv2.split(image)  # Split into individual channels

    channel_magnitudes = []
    for channel in channels:
        sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        channel_magnitudes.append(magnitude)

    # Sum the magnitudes across all channels
    combined_magnitude = np.sum(channel_magnitudes, axis=0)

    # Normalize to 0-255 range
    normalized_magnitude = cv2.normalize(combined_magnitude, dst=np.zeros_like(combined_magnitude), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized_magnitude.astype(np.uint8)
