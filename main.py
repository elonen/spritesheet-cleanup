#!/usr/bin/env python3
"""
Pixel Art Restoration Tool

Processes scaled-up, possibly squeezed pixel art with JPEG artifacts and noise,
restoring it to its original resolution.

If the image has an alpha channel, it is by default assumed to be a sprite sheet
and will be segmented into individual sprites. Grid is fine tuned for each sprite,
so the program can also handle images where the sprites are not precisely aligned
to a commong pixel grid - as long as their pixels are the roughly the same.

The program uses bilateral filtering to reduce noise, estimates the pixel grid,
segments the image into sprites, and restores each sprite to its original pixel size,
using median color for each pixel.
"""

import os
from pathlib import Path
import click
import cv2
import numpy as np

from alpha_processing import clean_alpha_channel
from grid_detection import estimate_grid_size, refine_grid
from sprite_segmentation import segment_sprites
from pixel_restoration import restore_smallscale_image
from grid_visualization import visualize_grid


@click.command(context_settings=dict(show_default=True))
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--min-sprite-size', '-m', type=float, default=2.0,
              help='Minimum size of sprite in pixels after restoration')
@click.option('--pixel-w-guess', '-w', type=float, help='Initial guess for pixel width')
@click.option('--pixel-h-guess', '-h', type=float, help='Initial guess for pixel height')
@click.option('--pixel-w-slop', '-ws', type=float, default=0.33,
              help='Multiplier for pixel width guess tolerance')
@click.option('--pixel-h-slop', '-hs', type=float, default=0.33,
              help='Multiplier for pixel height guess tolerance')
@click.option('--no-segment', '-n', is_flag=True, help='Skip sprite segmentation, restore the entire image')
@click.option('--bilateral-filter', '-b', is_flag=True, help='Apply bilateral noise filter')
@click.option('--debug', '-d', is_flag=True, help='Save intermediate images for debugging')
def main(input_path: str, output_path: str, min_sprite_size: float, pixel_w_guess: float | None,
         pixel_h_guess: float | None, pixel_w_slop: float, pixel_h_slop: float,
         no_segment: bool, bilateral_filter: bool, debug: bool) -> None:
    """Restore scaled-up, noisy, distorted pixel images to their original (lower) resolution.

    If the image has an alpha channel, it is assumed to be a sprite sheet and will by default
    be segmented into individual sprites. This can enhance the restoration process,
    especially if the sprites are not precisely aligned to a common pixel grid.

    INPUT_PATH is the path to the input image file.

    OUTPUT_PATH is the path where output images will be saved.

    If grid size is detected incorrectly, you can provide an initial guess for the pixel width
    and height using the -w -h options. The program will then
    estimate the pixel size based on this guess and the specified tolerance multipliers.
    """
    # Load the image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        click.echo(f"Error: Could not load image from {input_path}", err=True)
        return
    click.echo(f"Loaded image with shape {img.shape}")

    # Apply bilateral filter to each channel to reduce noise
    if bilateral_filter:
        for i in range(img.shape[2]):
            img[:, :, i] = cv2.bilateralFilter(img[:, :, i], d=5, sigmaColor=75, sigmaSpace=75)

    # Check if the image has an alpha channel, add one if it doesn't
    if img.shape[2] == 3:
        click.echo("No alpha channel detected, assuming the entire image is a sprite")
        alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    else:
        alpha = img[:, :, 3].copy()

    # Clean the alpha channel to remove noise
    cleaned_alpha = clean_alpha_channel(alpha)
    img[:, :, 3] = cleaned_alpha

    debug_dir = None
    if debug:
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / "01_cleaned_alpha.png"), cleaned_alpha)
        cv2.imwrite(str(debug_dir / "01_image_with_cleaned_alpha.png"), img)
        print("Debug images saved to 'debug' directory")

    # Estimate pixel size for the entire image
    pix_w, pix_h, w_std, h_std = estimate_grid_size(img, debug=debug,
        w_guess=pixel_w_guess, h_guess=pixel_h_guess,
        w_slop_mult=pixel_w_slop, h_slop_mult=pixel_h_slop)

    click.echo(f"Estimated global pixel size: {pix_w:.1f} x {pix_h:.1f} pixels")

    # Segment sprites using the minimum size derived from estimated pixel size
    if no_segment:
        sprite_regions = [(0, img.shape[0], 0, img.shape[1])]
        click.echo("Skipping sprite segmentation, processing the entire image")
    else:
        min_size = int((pix_w * pix_w) * min_sprite_size)
        click.echo(f"Minimum sprite size threshold: {min_size} pixels (before scaledown)")

        sprite_regions = segment_sprites(cleaned_alpha, min_size)
        click.echo(f"Detected {len(sprite_regions)} sprites")

        if debug_dir:
            segment_sprites_img = img.copy()
            for region in sprite_regions:
                y1, y2, x1, x2 = region
                cv2.rectangle(segment_sprites_img, (x1, y1), (x2, y2), (0, 255, 0, 255), 1)
            cv2.imwrite(str(debug_dir / "02_segmented.png"), segment_sprites_img)

    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each sprite, restore it to original pixel size, and save
    for i, region in enumerate(sprite_regions):
        y1, y2, x1, x2 = region
        sprite = img[y1:y2, x1:x2].copy()

        h_lines, v_lines, _edges_img = refine_grid(sprite, pix_w, pix_h, w_std, h_std)

        # Visualize the detected grid
        if debug_dir:
            cv2.imwrite(str(debug_dir / f"03_sprite_{i}.png"), sprite)

            grid_vis = visualize_grid(sprite, pix_w, pix_h, h_lines, v_lines)
            cv2.imwrite(str(debug_dir / f"03_grid_visualization_{i}.png"), grid_vis)

        restored_sprite = restore_smallscale_image(sprite, h_lines, v_lines)

        if debug_dir:
            # Visualize the restored sprite in (near) original resolution
            mean_pix_w = int(np.mean(np.diff(v_lines)) + 0.5)
            mean_pix_h = int(np.mean(np.diff(h_lines)) + 0.5)
            upscaled_sprite = cv2.resize(
                restored_sprite,
                (mean_pix_w*len(v_lines), mean_pix_h*len(h_lines)),
                interpolation=cv2.INTER_NEAREST
            )
            cv2.imwrite(str(debug_dir / f"04_restored_sprite_{i}.png"), upscaled_sprite)

        # Save individual sprite
        sprite_filename = f"{Path(output_path).stem}_sprite_{i}.png"
        sprite_path = os.path.join(output_dir, sprite_filename)
        cv2.imwrite(sprite_path, restored_sprite)

    click.echo(f"Sprites saved to {output_dir}")


if __name__ == "__main__":
    main()
