"""Image validation and quality checks."""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageFilter, ImageStat

SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "BMP"}


def validate_image(
    data: bytes,
    filename: str,
    max_bytes: int,
) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
    """Validate an uploaded image.

    Returns (PIL image or None, validation_dict).
    """
    warnings: List[Dict[str, str]] = []
    result: Dict[str, Any] = {
        "valid": False,
        "width": None,
        "height": None,
        "format": None,
        "mode": None,
        "file_size_bytes": len(data),
        "warnings": warnings,
        "error": None,
    }

    if not data:
        result["error"] = "No file was uploaded."
        return None, result

    if len(data) > max_bytes:
        max_mb = max_bytes / (1024 * 1024)
        result["error"] = (
            f"File is too large ({len(data) / (1024 * 1024):.1f} MB). "
            f"Maximum allowed size is {max_mb:.0f} MB."
        )
        return None, result

    # Extension check
    ext = (filename or "").rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""
    if ext and ext not in {"jpg", "jpeg", "png", "webp", "bmp"}:
        result["error"] = f"Unsupported file type '.{ext}'. Use JPG, JPEG, PNG, or WEBP."
        return None, result

    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # force decode to detect corruption
    except Exception:
        result["error"] = "The file could not be read as an image. It may be corrupted."
        return None, result

    fmt = (img.format or "").upper()
    if fmt and fmt not in SUPPORTED_FORMATS and fmt not in {"JPG"}:
        # Pillow reports JPEG as JPEG; allow anyway if it decoded
        if fmt not in {"JPEG", "PNG", "WEBP", "BMP"}:
            result["error"] = f"Unsupported image format '{fmt}'. Use JPG, JPEG, PNG, or WEBP."
            return None, result

    width, height = img.size
    result["width"] = width
    result["height"] = height
    result["format"] = fmt or "UNKNOWN"
    result["mode"] = img.mode
    result["valid"] = True

    # --- Quality warnings (never hard-reject) ---
    if width < 64 or height < 64:
        warnings.append(
            {
                "code": "very_small",
                "message": f"Image is very small ({width}×{height}px). Prediction quality may be limited.",
                "severity": "warning",
            }
        )
    elif width < 160 or height < 160:
        warnings.append(
            {
                "code": "small",
                "message": f"Image resolution is low ({width}×{height}px). A clearer image may improve results.",
                "severity": "warning",
            }
        )

    # Brightness / darkness
    try:
        gray = img.convert("L")
        stat = ImageStat.Stat(gray)
        mean_brightness = float(stat.mean[0]) if stat.mean else 128.0
        if mean_brightness < 40:
            warnings.append(
                {
                    "code": "too_dark",
                    "message": "Image appears very dark. Prediction may be affected.",
                    "severity": "warning",
                }
            )
        elif mean_brightness > 220:
            warnings.append(
                {
                    "code": "too_bright",
                    "message": "Image appears overexposed / very bright. Prediction may be affected.",
                    "severity": "warning",
                }
            )

        # Blur detection via variance of Laplacian-like edges
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges)
        edge_var = float(edge_stat.stddev[0]) if edge_stat.stddev else 0.0
        if edge_var < 8.0:
            warnings.append(
                {
                    "code": "blurry",
                    "message": "Image may be blurry. Prediction may be affected.",
                    "severity": "warning",
                }
            )
    except Exception:
        pass

    if any(w["severity"] == "warning" for w in warnings):
        # add a generic note without removing validity
        warnings.append(
            {
                "code": "quality_note",
                "message": "Image quality may affect prediction.",
                "severity": "info",
            }
        )

    return img, result


def load_image_rgb(img: Image.Image):
    """Convert PIL image to RGB numpy array."""
    import numpy as np

    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)
