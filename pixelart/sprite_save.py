#!/usr/bin/env python3
"""
Functions for saving pixel art sprites as individual images or as a spritesheet.
"""

import os
from pathlib import Path
import cv2
import numpy as np


def save_individual_sprites(
    sprites: list[np.ndarray],
    output_path: str,
    debug_dir: Path | None = None
) -> None:
    """
    Save each sprite as an individual file.

    Args:
        sprites: List of sprite images (NumPy arrays)
        output_path: Base path for the output files
        debug_dir: Directory for debug images (optional)
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sprite in enumerate(sprites):
        # Save individual sprite
        sprite_filename = f"{Path(output_path).stem}_sprite_{i}.png"
        sprite_path = os.path.join(output_dir, sprite_filename)
        cv2.imwrite(sprite_path, sprite)

        # Save debug visualization if needed
        if debug_dir:
            cv2.imwrite(str(debug_dir / f"04_restored_sprite_{i}.png"), sprite)


def create_spritesheet(
    sprites: list[np.ndarray],
    output_path: str,
    border_size: int = 2,
    debug_dir: Path | None = None
) -> None:
    """
    Create a single spritesheet containing all sprites with transparent borders.

    Args:
        sprites: List of sprite images (NumPy arrays)
        output_path: Path for the output spritesheet
        border_size: Size of the transparent border between sprites (default: 2)
        debug_dir: Directory for debug images (optional)
    """
    if not sprites:
        return

    # Determine the maximum width and height of sprites
    max_width = max(sprite.shape[1] for sprite in sprites)
    max_height = max(sprite.shape[0] for sprite in sprites)

    # Calculate number of sprites per row for a reasonable aspect ratio
    # Aim for a roughly square spritesheet
    num_sprites = len(sprites)
    num_cols = int(np.sqrt(num_sprites))
    num_rows = (num_sprites + num_cols - 1) // num_cols  # Ceiling division

    # Calculate spritesheet dimensions including borders
    sheet_width = num_cols * (max_width + border_size) + border_size
    sheet_height = num_rows * (max_height + border_size) + border_size

    # Create empty spritesheet with alpha channel (BGRA)
    spritesheet = np.zeros((sheet_height, sheet_width, 4), dtype=np.uint8)

    # Place sprites in the spritesheet
    for i, sprite in enumerate(sprites):
        row = i // num_cols
        col = i % num_cols

        # Calculate position in the spritesheet
        y_pos = row * (max_height + border_size) + border_size
        x_pos = col * (max_width + border_size) + border_size

        # Calculate centered position for sprites smaller than max dimensions
        y_offset = (max_height - sprite.shape[0]) // 2
        x_offset = (max_width - sprite.shape[1]) // 2

        # Copy sprite to spritesheet
        spritesheet[
            y_pos + y_offset:y_pos + y_offset + sprite.shape[0],
            x_pos + x_offset:x_pos + x_offset + sprite.shape[1]
        ] = sprite

    # Save spritesheet
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    spritesheet_path = os.path.join(output_dir, f"{Path(output_path).stem}_spritesheet.png")
    cv2.imwrite(spritesheet_path, spritesheet)

    # Save debug visualization if needed
    if debug_dir:
        cv2.imwrite(str(debug_dir / "05_spritesheet.png"), spritesheet)


def save_sprites(
    sprites: list[np.ndarray],
    output_path: str,
    create_sheet: bool = False,
    border_size: int = 2,
    debug_dir: Path | None = None
) -> None:
    """
    Save sprites either as individual files or as a spritesheet.

    Args:
        sprites: List of sprite images (NumPy arrays)
        output_path: Base path for output
        create_sheet: If True, create a spritesheet instead of individual files
        border_size: Size of transparent border in spritesheet (default: 2)
        debug_dir: Directory for debug images (optional)
    """
    if create_sheet:
        create_spritesheet(sprites, output_path, border_size, debug_dir)
    else:
        save_individual_sprites(sprites, output_path, debug_dir)
