"""
Pixel Art Restoration Tool

Processes scaled-up, possibly squeezed pixel art with JPEG artifacts and noise,
restoring it to its original resolution.

Public API:
    - process_spritesheet: Main generator function to process sprite sheets
    - ProcessedImage: Result object containing images with metadata
    - SpriteResult: Backwards compatibility alias (deprecated)
"""

from pixelart.api import process_spritesheet, ProcessedImage, SpriteResult

__version__ = "0.1.0"
__all__ = ["process_spritesheet", "ProcessedImage", "SpriteResult", "__version__"]
