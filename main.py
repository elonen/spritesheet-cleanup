#!/usr/bin/env python3
"""
Pixel Art Restoration Tool - Main Entry Point

This script processes scaled-up pixel art with JPEG artifacts and noise,
restoring it to its original pixelated style and resolution.
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np

from alpha_processing import clean_alpha_channel
from grid_detection import detect_grid_parameters
from sprite_segmentation import segment_sprites
from pixel_restoration import restore_pixelated_style

def main() -> None:
    parser = argparse.ArgumentParser(description="Restore scaled-up pixelated images")
    parser.add_argument("input_path", type=str, help="Path to the input image")
    parser.add_argument("output_path", type=str, help="Path to save the output image")
    parser.add_argument("--scale", type=float, default=None,
                        help="Known scale factor (if available)")
    parser.add_argument("--min-size-factor", type=float, default=0.75,
                        help="Minimum sprite size as a factor of pixel size squared")
    parser.add_argument("--sample-area", type=float, default=0.25,
                        help="Fraction of the image to sample for preliminary grid detection")
    parser.add_argument("--debug", action="store_true",
                        help="Save intermediate images for debugging")

    args = parser.parse_args()

    # Load the image
    img = cv2.imread(args.input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not load image from {args.input_path}")
        return

    # Check if the image has an alpha channel, add one if it doesn't
    if img.shape[2] == 3:
        print("No alpha channel detected, assuming the entire image is a sprite")
        alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    else:
        alpha = img[:, :, 3].copy()

    print(f"Loaded image with shape {img.shape}")

    # Step 1: Clean up the alpha channel
    cleaned_alpha = clean_alpha_channel(alpha)
    img[:, :, 3] = cleaned_alpha

    debug_dir = None
    if args.debug:
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / "01_cleaned_alpha.png"), cleaned_alpha)
        cv2.imwrite(str(debug_dir / "01_image_with_cleaned_alpha.png"), img)

    # Step 1.5: Preliminary grid detection on a sample area
    sample_size = int(min(img.shape[0], img.shape[1]) * args.sample_area)
    center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
    sample_y1 = max(0, center_y - sample_size // 2)
    sample_y2 = min(img.shape[0], center_y + sample_size // 2)
    sample_x1 = max(0, center_x - sample_size // 2)
    sample_x2 = min(img.shape[1], center_x + sample_size // 2)

    sample_img = img[sample_y1:sample_y2, sample_x1:sample_x2].copy()

    if debug_dir:
        cv2.imwrite(str(debug_dir / "01_5_sample_area.png"), sample_img)

    # Detect grid parameters from the sample
    pixel_size = args.scale if args.scale is not None else detect_grid_parameters(sample_img)
    print(f"Estimated pixel size: {pixel_size}")

    # Step 2 & 3: Segment sprites using the minimum size derived from pixel size
    min_size = int((pixel_size * pixel_size) * args.min_size_factor)
    print(f"Minimum sprite size threshold: {min_size} pixels")

    sprite_regions = segment_sprites(cleaned_alpha, min_size)
    print(f"Detected {len(sprite_regions)} sprites")

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 4 & 5: Process each sprite, restore pixelated style, and save
    processed_sprites = []

    for i, region in enumerate(sprite_regions):
        y1, y2, x1, x2 = region
        sprite = img[y1:y2, x1:x2].copy()

        if debug_dir:
            cv2.imwrite(str(debug_dir / f"03_sprite_{i}.png"), sprite)

        # Detect pixel grid more precisely for this specific sprite
        sprite_pixel_size = detect_grid_parameters(sprite, pixel_size)

        # Restore pixelated style
        restored_sprite = restore_pixelated_style(sprite, sprite_pixel_size)

        if debug_dir:
            cv2.imwrite(str(debug_dir / f"04_restored_sprite_{i}.png"), restored_sprite)

        # Scale down to original resolution
        original_height = max(1, int((y2 - y1) / sprite_pixel_size + 0.5))
        original_width = max(1, int((x2 - x1) / sprite_pixel_size + 0.5))

        scaled_sprite = cv2.resize(
            restored_sprite,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST
        )

        if debug_dir:
            cv2.imwrite(str(debug_dir / f"05_scaled_sprite_{i}.png"), scaled_sprite)

        # Save individual sprite
        sprite_filename = f"{Path(args.output_path).stem}_sprite_{i}.png"
        sprite_path = os.path.join(output_dir, sprite_filename)
        cv2.imwrite(sprite_path, scaled_sprite)

        processed_sprites.append({
            "sprite": scaled_sprite,
            "original_position": (x1, y1, x2, y2),
            "scaled_position": (
                int(x1 / sprite_pixel_size),
                int(y1 / sprite_pixel_size),
                int(x2 / sprite_pixel_size),
                int(y2 / sprite_pixel_size)
            )
        })

    print(f"Successfully processed {len(processed_sprites)} sprites")
    print(f"Sprites saved to {output_dir}")

if __name__ == "__main__":
    main()
