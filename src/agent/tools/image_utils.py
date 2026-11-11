"""Image utility helpers for multimodal preprocessing and encoding."""

import base64
import io
from typing import Dict, Optional, Tuple
from PIL import Image


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load PIL Image from raw bytes."""
    return Image.open(io.BytesIO(data))


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to base64 data string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_image(base64_str: str) -> Image.Image:
    """Convert a base64 string back to PIL Image."""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))


def get_image_metadata(data: bytes) -> Dict[str, any]:
    """Extract format, size, and dimensions from raw image bytes."""
    try:
        img = Image.open(io.BytesIO(data))
        return {
            "format": img.format or "PNG",
            "size_bytes": len(data),
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
        }
    except Exception as e:
        return {"error": str(e), "size_bytes": len(data)}


def resize_image_if_needed(
    img: Image.Image, max_dim: int = 1536
) -> Tuple[Image.Image, bool]:
    """Resize image to fit within max dimension while maintaining aspect ratio."""
    width, height = img.size
    if width <= max_dim and height <= max_dim:
        return img, False

    if width > height:
        new_width = max_dim
        new_height = int(height * (max_dim / width))
    else:
        new_height = max_dim
        new_width = int(width * (max_dim / height))

    resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized, True

