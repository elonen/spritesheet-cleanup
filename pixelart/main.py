#!/usr/bin/env python3
"""
Pixel Art Restoration Tool - Command Line Interface

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

from pathlib import Path
import click
import cv2

from pixelart.api import process_spritesheet
from pixelart.sprite_save import save_sprites


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
@click.option('--spritesheet', '-s', is_flag=True,
              help='Create a single spritesheet instead of individual files')
@click.option('--debug', '-d', is_flag=True, help='Save intermediate images for debugging')
def main(input_path: str, output_path: str, min_sprite_size: float, pixel_w_guess: float | None,
         pixel_h_guess: float | None, pixel_w_slop: float, pixel_h_slop: float,
         no_segment: bool, bilateral_filter: bool, spritesheet: bool, debug: bool) -> None:
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

    # Setup debug directory if needed
    debug_dir = None
    if debug:
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        click.echo("Debug mode enabled, saving intermediate images to 'debug' directory")

    # Process the image using the library generator API
    try:
        # Collect sprites and handle debug images
        restored_sprites = []
        num_debug_images = 0

        for result in process_spritesheet(
            img,
            min_sprite_size=min_sprite_size,
            pixel_w_guess=pixel_w_guess,
            pixel_h_guess=pixel_h_guess,
            pixel_w_slop=pixel_w_slop,
            pixel_h_slop=pixel_h_slop,
            no_segment=no_segment,
            bilateral_filter=bilateral_filter,
            debug=debug
        ):
            if result.is_debug:
                # Save debug images to debug directory
                if debug_dir:
                    # Determine file extension based on number of channels
                    debug_path = debug_dir / f"{result.name}.png"
                    cv2.imwrite(str(debug_path), result.image)
                    num_debug_images += 1
            else:
                # Collect sprites for final output
                restored_sprites.append(result.image)

    except ValueError as e:
        click.echo(f"Error processing image: {e}", err=True)
        return

    click.echo(f"Processed {len(restored_sprites)} sprite(s)")
    if debug:
        click.echo(f"Saved {num_debug_images} debug image(s) to {debug_dir}")

    # Save sprites using the save function
    save_sprites(
        restored_sprites,
        output_path,
        create_sheet=spritesheet,
        debug_dir=None  # Debug images already saved above
    )

    if spritesheet:
        click.echo(f"Spritesheet saved to {Path(output_path).parent}")
    else:
        click.echo(f"Sprites saved to {Path(output_path).parent}")


if __name__ == "__main__":
    main()
