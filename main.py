#!/usr/bin/env python3
"""
Pixel Art Restoration Tool

Processes scaled-up, possibly squeezed pixel art with JPEG artifacts and noise,
restoring it to its original resolution.

If the image has an alpha channel, it is by default assumed to be a sprite sheet
and will be segmented into individual sprites.

The program uses bilateral filtering to reduce noise, estimates the pixel grid,
segments the image into sprites, and restores each sprite to its original pixel size,
using median color for each pixel.
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np

from alpha_processing import clean_alpha_channel
from grid_detection import estimate_grid_size, refine_grid
from sprite_segmentation import segment_sprites
from pixel_restoration import restore_smallscale_image
from grid_visualization import visualize_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore scaled-up pixelated images")
    parser.add_argument("input_path", type=str, help="Path to the input image")
    parser.add_argument("output_path", type=str, help="Path to save the output image")

    parser.add_argument("--scale", type=float, default=None,
                        help="Known scale factor (if available)")

    parser.add_argument("--min-sprite-size", type=float, default=2.0,
                        help="For segmenting sprites, minimum size of sprite in pixels after restoration")

    parser.add_argument("--pixel-w-guess", type=float, default=None,
                        help="Initial guess for pixel width")
    parser.add_argument("--pixel-h-guess", type=float, default=None,
                        help="Initial guess for pixel height")
    parser.add_argument("--pixel-w-slop", type=float, default=0.5,
                        help="Multiplier for pixel width guess. Deviations larger than this will be discarded.")
    parser.add_argument("--pixel-h-slop", type=float, default=0.5,
                        help="Multiplier for pixel height guess. Deviations larger than this will be discarded.")

    parser.add_argument("--no-segment", action="store_true",
                        help="Skip sprite segmentation, restore the entire image")

    parser.add_argument("--bilateral-filter", action="store_true",
                        help="Apply bilateral noise filter")
    parser.add_argument("--debug", action="store_true",
                        help="Save intermediate images for debugging")

    args = parser.parse_args()

    # Load the image
    img = cv2.imread(args.input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not load image from {args.input_path}")
        return
    print(f"Loaded image with shape {img.shape}")

    # Apply bilateral filter to each channel to reduce noise
    if args.bilateral_filter:
        for i in range(img.shape[2]):
            img[:, :, i] = cv2.bilateralFilter(img[:, :, i], d=5, sigmaColor=75, sigmaSpace=75)

    # Check if the image has an alpha channel, add one if it doesn't
    if img.shape[2] == 3:
        print("No alpha channel detected, assuming the entire image is a sprite")
        alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    else:
        alpha = img[:, :, 3].copy()


    # Clean the alpha channel to remove noise
    cleaned_alpha = clean_alpha_channel(alpha)
    img[:, :, 3] = cleaned_alpha

    debug_dir = None
    if args.debug:
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / "01_cleaned_alpha.png"), cleaned_alpha)
        cv2.imwrite(str(debug_dir / "01_image_with_cleaned_alpha.png"), img)


    # Estimate pixel size for the entire image
    pix_w, pix_h, w_std, h_std = estimate_grid_size(img, debug=args.debug,
        w_guess=args.pixel_w_guess, h_guess=args.pixel_h_guess,
        w_slop_mult=args.pixel_w_slop, h_slop_mult=args.pixel_h_slop)

    print(f"Estimated global pixel size: {pix_w:.1f} x {pix_h:.1f} pixels")


    # Segment sprites using the minimum size derived from estimated pixel size
    if args.no_segment:
        sprite_regions = [(0, img.shape[0], 0, img.shape[1])]
        print("Skipping sprite segmentation, processing the entire image")
    else:
        min_size = int((pix_w * pix_w) * args.min_sprite_size)
        print(f"Minimum sprite size threshold: {min_size} pixels")

        sprite_regions = segment_sprites(cleaned_alpha, min_size)
        print(f"Detected {len(sprite_regions)} sprites")

        if debug_dir:
            segment_sprites_img = img.copy()
            for region in sprite_regions:
                y1, y2, x1, x2 = region
                print(region)
                cv2.rectangle(segment_sprites_img, (x1, y1), (x2, y2), (0, 255, 0, 255), 1)
            cv2.imwrite(str(debug_dir / "02_segmented.png"), segment_sprites_img)


    # Create output directory if it doesn't exist
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)


    # Process each sprite, restore it to original pixel size, and save
    for i, region in enumerate(sprite_regions):
        y1, y2, x1, x2 = region
        sprite = img[y1:y2, x1:x2].copy()

        h_lines, v_lines = refine_grid(sprite, pix_w, pix_h, w_std, h_std)

        # Visualize the detected grid
        if debug_dir:
            cv2.imwrite(str(debug_dir / f"03_sprite_{i}.png"), sprite)

            # Standard grid visualization
            grid_vis = visualize_grid(sprite, pix_w, pix_h, h_lines, v_lines)
            cv2.imwrite(str(debug_dir / f"03_grid_visualization_{i}.png"), grid_vis)

        restored_sprite = restore_smallscale_image(sprite, h_lines, v_lines)

        if debug_dir:
            # Visualize the restored sprite in (near) original resolution
            mean_pix_w = int(np.mean(np.diff(v_lines)) + 0.5)
            mean_pix_h = int(np.mean(np.diff(h_lines)) + 0.5)
            print("Mean pixel size: ", mean_pix_w, mean_pix_h)
            upscaled_sprite = cv2.resize(
                restored_sprite,
                (mean_pix_w*len(v_lines), mean_pix_h*len(h_lines)),
                interpolation=cv2.INTER_NEAREST
            )
            print("upscaled_sprite sprite shape: ", upscaled_sprite.shape)
            cv2.imwrite(str(debug_dir / f"04_restored_sprite_{i}.png"), upscaled_sprite)

        # Save individual sprite
        sprite_filename = f"{Path(args.output_path).stem}_sprite_{i}.png"
        sprite_path = os.path.join(output_dir, sprite_filename)
        cv2.imwrite(sprite_path, restored_sprite)

    print(f"Sprites saved to {output_dir}")

if __name__ == "__main__":
    main()
