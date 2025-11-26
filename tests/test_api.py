"""
End-to-end tests for the spritesheet_cleanup library API.

Tests that the library can be used programmatically to process images
without using the CLI.
"""

import numpy as np
import cv2
import pytest
from pathlib import Path

from spritesheet_cleanup import process_spritesheet, ProcessedImage


def test_process_spritesheet_basic():
    """Test basic spritesheet processing with raw numpy array."""
    # Load example image as numpy array (simulating library usage)
    example_path = Path(__file__).parent.parent / "examples" / "example-input.png"
    image = cv2.imread(str(example_path), cv2.IMREAD_UNCHANGED)

    assert image is not None, "Failed to load example image"
    assert image.shape[2] in (3, 4), "Image should be BGR or BGRA"

    # Process the image using the library generator API (not CLI)
    results = list(process_spritesheet(image))

    # Verify we got results
    assert len(results) > 0, "Should extract at least one sprite"

    # Separate sprites from debug images
    sprites = [r for r in results if not r.is_debug]
    debug_images = [r for r in results if r.is_debug]

    # Without debug flag, should only get sprites
    assert len(debug_images) == 0, "Should not get debug images without debug=True"
    assert len(sprites) > 0, "Should extract at least one sprite"

    # Verify each sprite is a ProcessedImage with correct structure
    for result in sprites:
        assert isinstance(result, ProcessedImage), "Result should be ProcessedImage"

        # Check image is valid numpy array with BGRA format
        assert isinstance(result.image, np.ndarray), "Sprite image should be numpy array"
        assert result.image.dtype == np.uint8, "Sprite should be uint8"
        assert result.image.shape[2] == 4, "Sprite should be BGRA (4 channels)"
        assert result.image.shape[0] > 0, "Sprite should have height"
        assert result.image.shape[1] > 0, "Sprite should have width"

        # Check bbox has correct structure (y1, y2, x1, x2)
        assert isinstance(result.bbox, tuple), "bbox should be tuple"
        assert len(result.bbox) == 4, "bbox should have 4 coordinates"
        y1, y2, x1, x2 = result.bbox
        # Coordinates might be numpy ints, so just check they're numeric
        assert 0 <= y1 < y2, "y1 should be less than y2"
        assert 0 <= x1 < x2, "x1 should be less than x2"
        assert y2 <= image.shape[0], "y2 should be within image height"
        assert x2 <= image.shape[1], "x2 should be within image width"

        # Check name
        assert result.name.startswith("sprite_"), "Sprite name should start with 'sprite_'"
        assert not result.is_debug, "Sprites should not be marked as debug"

    print(f"Successfully extracted {len(sprites)} sprites")
    for i, result in enumerate(sprites):
        h, w = result.image.shape[:2]
        y1, y2, x1, x2 = result.bbox
        print(f"  {result.name}: {w}x{h} pixels, from bbox ({y1}, {y2}, {x1}, {x2})")


def test_process_spritesheet_with_options():
    """Test spritesheet processing with custom options."""
    example_path = Path(__file__).parent.parent / "examples" / "example-input.png"
    image = cv2.imread(str(example_path), cv2.IMREAD_UNCHANGED)

    # Test with bilateral filter
    results = list(process_spritesheet(
        image,
        bilateral_filter=True,
        min_sprite_size=2.0,
        pixel_w_slop=0.5,
        pixel_h_slop=0.5
    ))

    sprites = [r for r in results if not r.is_debug]
    assert len(sprites) > 0, "Should extract sprites with custom options"


def test_process_spritesheet_no_segment():
    """Test processing entire image without segmentation."""
    example_path = Path(__file__).parent.parent / "examples" / "example-input.png"
    image = cv2.imread(str(example_path), cv2.IMREAD_UNCHANGED)

    # Process without segmentation
    results = list(process_spritesheet(image, no_segment=True))

    # Separate sprites from any potential debug images
    sprites = [r for r in results if not r.is_debug]

    # Should return exactly one sprite (the entire image)
    assert len(sprites) == 1, "no_segment should return single sprite"

    result = sprites[0]
    y1, y2, x1, x2 = result.bbox

    # Bbox should cover the entire image
    assert y1 == 0, "Should start at top"
    assert x1 == 0, "Should start at left"
    assert y2 == image.shape[0], "Should end at bottom"
    assert x2 == image.shape[1], "Should end at right"


def test_process_spritesheet_bgr_input():
    """Test that BGR images (without alpha) are handled correctly."""
    example_path = Path(__file__).parent.parent / "examples" / "example-input.png"
    image = cv2.imread(str(example_path), cv2.IMREAD_COLOR)  # Load as BGR (no alpha)

    assert image.shape[2] == 3, "Image should be BGR (3 channels)"

    # Should work with BGR images too
    results = list(process_spritesheet(image, no_segment=True))

    sprites = [r for r in results if not r.is_debug]
    assert len(sprites) > 0, "Should process BGR images"
    assert sprites[0].image.shape[2] == 4, "Output should still be BGRA"


def test_process_spritesheet_invalid_input():
    """Test that invalid inputs raise appropriate errors."""
    with pytest.raises(ValueError, match="image.*None"):
        list(process_spritesheet(None))  # type: ignore

    with pytest.raises(ValueError, match="shape"):
        list(process_spritesheet(np.zeros((10, 10), dtype=np.uint8)))  # 2D array, needs 3D


def test_sprite_result_in_order():
    """Test that sprites are returned in order they appear in source."""
    example_path = Path(__file__).parent.parent / "examples" / "example-input.png"
    image = cv2.imread(str(example_path), cv2.IMREAD_UNCHANGED)

    results = list(process_spritesheet(image))
    sprites = [r for r in results if not r.is_debug]

    # Sprites should be ordered by their position in the original image
    # This typically means top-to-bottom, left-to-right ordering
    for i in range(len(sprites) - 1):
        curr_y1 = sprites[i].bbox[0]
        next_y1 = sprites[i + 1].bbox[0]
        # Each sprite should start at or below the previous one's start
        # (or to the right if at same height)
        assert curr_y1 <= next_y1 + image.shape[0] // 2, "Sprites should be reasonably ordered"


def test_process_spritesheet_debug_mode():
    """Test that debug mode yields debug images."""
    example_path = Path(__file__).parent.parent / "examples" / "example-input.png"
    image = cv2.imread(str(example_path), cv2.IMREAD_UNCHANGED)

    # Process with debug enabled
    results = list(process_spritesheet(image, debug=True))

    # Separate sprites and debug images
    sprites = [r for r in results if not r.is_debug]
    debug_images = [r for r in results if r.is_debug]

    # Should have both sprites and debug images
    assert len(sprites) > 0, "Should extract sprites"
    assert len(debug_images) > 0, "Should yield debug images when debug=True"

    # Check debug images have correct structure
    for debug_img in debug_images:
        assert isinstance(debug_img, ProcessedImage), "Debug image should be ProcessedImage"
        assert debug_img.is_debug, "Should be marked as debug"
        assert debug_img.name.startswith("debug_"), "Debug image name should start with 'debug_'"
        assert isinstance(debug_img.image, np.ndarray), "Debug image should be numpy array"

    print(f"Debug mode: {len(sprites)} sprites, {len(debug_images)} debug images")
