"""Image preprocessing utilities for efficient memory usage."""

import errno
import os
from pathlib import Path
from typing import Optional

from loguru import logger
from PIL import Image, UnidentifiedImageError


class ImageValidationError(Exception):
    """Raised when image validation fails with user-friendly message."""
    pass


def validate_image_file(path: str, warn_large: bool = True) -> tuple[bool, Optional[str]]:
    """Validate image file exists and is readable.

    Handles unicode, spaces, and special characters in file paths.

    Args:
        path: Path to image file (can contain unicode, spaces, etc.)
        warn_large: Whether to warn about very large images (default: True)

    Returns:
        Tuple of (is_valid, error_message)
        If valid: (True, None)
        If invalid: (False, "user-friendly error message")
    """
    # Convert to Path object and resolve to absolute path
    # This handles relative paths, symlinks, ~, and normalizes separators
    try:
        image_path = Path(path).expanduser().resolve()
    except (OSError, RuntimeError) as e:
        # Handle cases like circular symlinks, permission issues
        return False, f"Cannot resolve path: {path}\nError: {str(e)}"

    # Check if file exists
    if not image_path.exists():
        return False, f"Image file not found: {path}"

    # Check if it's a file (not directory)
    if not image_path.is_file():
        return False, f"Path is not a file: {path}"

    # Check file size (not 0 bytes)
    file_size = image_path.stat().st_size
    if file_size == 0:
        return False, f"Image file is empty (0 bytes): {path}"

    # Warn about extremely large files (>100MB likely indicates huge dimensions)
    if warn_large and file_size > 100 * 1024 * 1024:  # 100MB
        logger.warning(f"Very large image file ({file_size / (1024**2):.1f}MB): {path}")

    # Check file extension
    valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
    if image_path.suffix.lower() not in valid_extensions:
        return False, (
            f"Invalid image format: {image_path.suffix}\n"
            f"Supported formats: {', '.join(sorted(valid_extensions))}"
        )

    # Try to open and verify the image
    try:
        with Image.open(path) as img:
            # Verify the image can be read (loads header)
            img.verify()

            # Reopen to check dimensions (verify() closes the file)
            img = Image.open(path)
            width, height = img.size

            # Check for extremely large dimensions
            max_reasonable_dim = 20000  # 20k pixels on either side
            if width > max_reasonable_dim or height > max_reasonable_dim:
                return False, (
                    f"Image dimensions too large: {width}x{height}\n"
                    f"Maximum supported: {max_reasonable_dim}x{max_reasonable_dim}\n"
                    f"Please resize the image before processing."
                )

            # Warn about very large dimensions (but still processable)
            if warn_large and (width > 10000 or height > 10000):
                logger.warning(
                    f"Very large image dimensions: {width}x{height}. "
                    f"Processing may be slow and memory-intensive."
                )

        return True, None
    except UnidentifiedImageError:
        return False, f"File is not a valid image or format not supported: {path}"
    except Exception as e:
        return False, f"Image file appears corrupted: {path}\nError: {str(e)}"


def preprocess_image(path: str, max_dim: int = 1920) -> Image.Image:
    """Load and preprocess image with memory-efficient resizing.

    Args:
        path: Path to image file
        max_dim: Maximum dimension (width or height) in pixels

    Returns:
        Preprocessed PIL Image in RGB format

    Raises:
        ImageValidationError: If image is invalid with clear user-friendly message
    """
    # Validate image file first
    is_valid, error_msg = validate_image_file(path)
    if not is_valid:
        logger.error(f"Image validation failed: {error_msg}")
        raise ImageValidationError(error_msg)

    # Image.verify() closes the file, so we need to reopen
    img = Image.open(path)
    original_size = img.size
    original_mode = img.mode

    # Estimate memory usage (width * height * channels * bytes_per_channel)
    channels = len(img.getbands())
    estimated_mb = (original_size[0] * original_size[1] * channels * 4) / (1024 ** 2)  # 4 bytes for float32

    # For very large images (>50MB), use aggressive downsampling first
    if estimated_mb > 50:
        logger.info(f"Large image detected ({estimated_mb:.1f}MB). Using memory-efficient processing.")
        # Use thumbnail for in-place memory-efficient resizing
        img.thumbnail((max_dim, max_dim), Image.Resampling.BICUBIC)
        img = img.convert("RGB")
        logger.debug(f"Resized from {original_size} to {img.size}")
        return img

    # Convert to RGB if needed (RGBA, grayscale, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize if exceeds max dimension (bicubic interpolation for quality)
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.BICUBIC)
        logger.debug(f"Resized from {original_size} to {img.size}")

    return img


def load_image_efficiently(path: str, max_dim: int) -> Image.Image:
    """Load image with in-place thumbnail for minimal memory allocation.

    Use this for very large images where memory is constrained.

    Args:
        path: Path to image file
        max_dim: Maximum dimension (width or height) in pixels

    Returns:
        Preprocessed PIL Image in RGB format

    Raises:
        ImageValidationError: If image is invalid with clear user-friendly message
    """
    # Validate image file first
    is_valid, error_msg = validate_image_file(path)
    if not is_valid:
        logger.error(f"Image validation failed: {error_msg}")
        raise ImageValidationError(error_msg)

    # Image.verify() closes the file, so we need to reopen
    img = Image.open(path)  # Lazy load, only reads header

    if max(img.size) > max_dim:
        # Thumbnail modifies in-place, avoiding extra allocation
        img.thumbnail((max_dim, max_dim), Image.Resampling.BICUBIC)

    return img.convert("RGB")  # Only decode needed channels


def save_annotated_image(
    image: Image.Image,
    detections: list[dict],
    output_path: Path,
    original_size: tuple[int, int] | None = None
) -> Path:
    """Draw bounding boxes or points on image and save.

    Handles unicode and special characters in output paths.
    Automatically detects and converts normalized coordinates (0.0-1.0) to pixels.
    Handles VLM resolution scaling when image was resized before processing.

    VLM Resolution Scaling Problem:
        Vision-Language Models (VLMs) often internally resize images to standard sizes
        (224x224, 336x336, 448x448, 512x512, 768x768, 1024x1024, 1280x1280) before
        processing. When the VLM returns bounding box coordinates or point locations,
        these coordinates are based on the RESIZED image dimensions, not the original.

        Example:
            - Original image: 1920x1080 pixels
            - VLM internally resizes to: 336x336 pixels
            - VLM returns bbox: [50, 100, 150, 200] (based on 336x336)
            - To draw on original image: must scale by (1920/336, 1080/336)
            - Correct bbox: [285, 321, 857, 643] (on 1920x1080)

    Detection Strategy (Multi-Heuristic):
        This function uses multiple heuristics to detect when VLM resolution scaling
        is needed, as we don't always know what resolution the VLM actually used:

        1. Normalized coordinates (0.0-1.0): Convert to pixels
        2. Coordinates exceed image: Definite mismatch, match against common VLM sizes
        3. Small coords on large image: If image >1000px but coords <600, likely resized
        4. Aspect ratio mismatch: Square bbox on non-square image suggests VLM resizing

    Limitations:
        - Heuristics may fail if VLM uses unusual resolutions
        - Cannot detect scaling when coords are valid for both VLM and original sizes
        - Best solution: Modify providers to track actual VLM processing dimensions

    Args:
        image: Original PIL Image (at full resolution, not VLM-resized)
        detections: List of detection dicts with 'bbox' or 'coordinates' keys
                   - bbox can be pixel coords [x1, y1, x2, y2] or normalized [0.0-1.0]
                   - coordinates can be pixel {x, y} or normalized {x: 0.0-1.0, y: 0.0-1.0}
        output_path: Path to save annotated image (can contain unicode/spaces)
        original_size: Optional (width, height) of original image before VLM processing.
                      Currently unused but kept for future enhancements.

    Returns:
        Path where image was saved

    Raises:
        OSError: If path cannot be created or file cannot be saved

    Note:
        When VLM resolution mismatch is detected, warning logs are emitted showing
        the detected VLM resolution and scaling factors applied.
    """
    from PIL import ImageDraw

    # Normalize and resolve output path (handles ~, relative paths, etc.)
    try:
        output_path = Path(output_path).expanduser().resolve()
    except (OSError, RuntimeError) as e:
        logger.error(f"Invalid output path: {output_path}")
        raise OSError(f"Cannot resolve output path: {output_path}\nError: {str(e)}")

    # Create a copy to avoid modifying original
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    img_width, img_height = image.size

    def is_normalized(value: float) -> bool:
        """Check if a coordinate value is normalized (0.0-1.0)."""
        return 0.0 <= value <= 1.0

    def detect_vlm_resolution_mismatch(bbox: list) -> tuple[bool, float, float]:
        """Detect if bbox coordinates are from a different (VLM-resized) resolution.

        VLMs often resize images to standard sizes: 224, 336, 448, 512, 768, 1024.
        This function uses multiple heuristics to detect coordinate mismatches.

        Returns:
            (needs_scaling, scale_x, scale_y)
        """
        x1, y1, x2, y2 = bbox

        # Common VLM input resolutions
        common_vlm_sizes = [224, 336, 384, 448, 512, 768, 1024, 1280]

        # Maximum coordinate in the bbox
        max_coord = max(x2, y2)
        min_coord = min(x1, y1)

        # CASE 1: Coordinates exceed image dimensions (definite mismatch)
        if max_coord > max(img_width, img_height):
            # Try to match against common VLM sizes
            for vlm_size in common_vlm_sizes:
                # Allow small tolerance (within 10% of VLM size)
                if vlm_size * 0.9 <= max_coord <= vlm_size * 1.1:
                    scale_x = img_width / vlm_size
                    scale_y = img_height / vlm_size

                    logger.warning(
                        f"VLM resolution mismatch detected (coords exceed image): "
                        f"bbox appears to be for {vlm_size}x{vlm_size} but image is {img_width}x{img_height}. "
                        f"Scaling by ({scale_x:.2f}, {scale_y:.2f})"
                    )
                    return True, scale_x, scale_y

            # Coords exceed but don't match known size - use max_coord as assumed VLM size
            assumed_vlm_size = max_coord
            scale_x = img_width / assumed_vlm_size
            scale_y = img_height / assumed_vlm_size

            logger.warning(
                f"VLM resolution mismatch detected: coords ({max_coord}) exceed image size ({img_width}x{img_height}). "
                f"Assuming VLM resolution {assumed_vlm_size}x{assumed_vlm_size}, scaling by ({scale_x:.2f}, {scale_y:.2f})"
            )
            return True, scale_x, scale_y

        # CASE 2: Check if all coordinates are suspiciously small (heuristic for VLM resizing)
        # If image is large (>1000px) but all coords are < 600, likely VLM resized to small size
        if max(img_width, img_height) > 1000 and max_coord <= 600:
            # Try to match against common VLM sizes
            for vlm_size in common_vlm_sizes:
                # If max_coord is close to a common VLM size (within 80-110% range)
                # This handles cases like: image is 1920x1080, coords are [50,50,300,300] (for 336x336 VLM)
                if vlm_size * 0.8 <= max_coord <= vlm_size * 1.1:
                    scale_x = img_width / vlm_size
                    scale_y = img_height / vlm_size

                    logger.warning(
                        f"VLM resolution mismatch suspected (small coords on large image): "
                        f"image is {img_width}x{img_height}, bbox max is {max_coord}. "
                        f"Assuming VLM resized to {vlm_size}x{vlm_size}, scaling by ({scale_x:.2f}, {scale_y:.2f})"
                    )
                    return True, scale_x, scale_y

        # CASE 3: Check aspect ratio mismatch
        # If image has very different aspect ratio but coords suggest square processing
        image_aspect = img_width / img_height
        if abs(image_aspect - 1.0) > 0.3:  # Image is not square (e.g., 16:9)
            # If coords suggest square processing (similar x and y ranges)
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            if bbox_width > 0 and bbox_height > 0:
                bbox_aspect = bbox_width / bbox_height
                # Bbox is squarish (aspect close to 1.0) but image is not
                if abs(bbox_aspect - 1.0) < 0.3 and max_coord <= 1280:
                    # Check if max_coord matches a common VLM size
                    for vlm_size in common_vlm_sizes:
                        if vlm_size * 0.7 <= max_coord <= vlm_size * 1.1:
                            scale_x = img_width / vlm_size
                            scale_y = img_height / vlm_size

                            logger.warning(
                                f"VLM resolution mismatch suspected (aspect ratio): "
                                f"image {img_width}x{img_height} (aspect {image_aspect:.2f}) but bbox suggests square VLM processing. "
                                f"Assuming {vlm_size}x{vlm_size}, scaling by ({scale_x:.2f}, {scale_y:.2f})"
                            )
                            return True, scale_x, scale_y

        # CASE 4: No mismatch detected - coordinates appear correct
        logger.debug(f"No VLM resolution mismatch detected for bbox {bbox} on image {img_width}x{img_height}")
        return False, 1.0, 1.0

    def convert_bbox_to_pixels(bbox: list) -> list:
        """Convert bbox from normalized to pixel coordinates if needed."""
        x1, y1, x2, y2 = bbox

        # Check if all values are normalized
        if all(is_normalized(v) for v in bbox):
            logger.debug(f"Converting normalized bbox {bbox} to pixels")
            return [
                int(x1 * img_width),
                int(y1 * img_height),
                int(x2 * img_width),
                int(y2 * img_height)
            ]

        # Check for VLM resolution mismatch
        needs_scaling, scale_x, scale_y = detect_vlm_resolution_mismatch(bbox)

        if needs_scaling:
            return [
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y)
            ]

        # Already in correct pixel coordinates
        return [int(x1), int(y1), int(x2), int(y2)]

    def convert_point_to_pixels(x: float, y: float) -> tuple[int, int]:
        """Convert point from normalized to pixel coordinates if needed."""
        # CASE 1: Normalized coordinates (0.0-1.0)
        if is_normalized(x) and is_normalized(y):
            logger.debug(f"Converting normalized point ({x}, {y}) to pixels")
            return (int(x * img_width), int(y * img_height))

        # CASE 2: Point exceeds image dimensions (definite VLM mismatch)
        if x > img_width or y > img_height:
            # Assume coords are from VLM-resized image
            max_coord = max(x, y)
            common_vlm_sizes = [224, 336, 384, 448, 512, 768, 1024, 1280]

            for vlm_size in common_vlm_sizes:
                if vlm_size * 0.9 <= max_coord <= vlm_size * 1.1:
                    scale_x = img_width / vlm_size
                    scale_y = img_height / vlm_size
                    logger.warning(
                        f"VLM resolution mismatch detected for point (coords exceed image): "
                        f"({x}, {y}) appears to be for {vlm_size}x{vlm_size} but image is {img_width}x{img_height}. "
                        f"Scaling by ({scale_x:.2f}, {scale_y:.2f})"
                    )
                    return (int(x * scale_x), int(y * scale_y))

            # Unknown resolution, try to scale proportionally
            scale_x = img_width / max_coord
            scale_y = img_height / max_coord
            logger.warning(
                f"VLM resolution mismatch detected: point ({x}, {y}) exceeds image size ({img_width}x{img_height}). "
                f"Assuming VLM resolution {max_coord}x{max_coord}, scaling by ({scale_x:.2f}, {scale_y:.2f})"
            )
            return (int(x * scale_x), int(y * scale_y))

        # CASE 3: Check if coordinates are suspiciously small (heuristic for VLM resizing)
        max_coord = max(x, y)
        if max(img_width, img_height) > 1000 and max_coord <= 600:
            common_vlm_sizes = [224, 336, 384, 448, 512, 768, 1024, 1280]
            for vlm_size in common_vlm_sizes:
                if vlm_size * 0.8 <= max_coord <= vlm_size * 1.1:
                    scale_x = img_width / vlm_size
                    scale_y = img_height / vlm_size
                    logger.warning(
                        f"VLM resolution mismatch suspected for point (small coords on large image): "
                        f"image is {img_width}x{img_height}, point max is {max_coord}. "
                        f"Assuming VLM resized to {vlm_size}x{vlm_size}, scaling by ({scale_x:.2f}, {scale_y:.2f})"
                    )
                    return (int(x * scale_x), int(y * scale_y))

        # CASE 4: No mismatch detected - coordinates appear correct
        logger.debug(f"No VLM resolution mismatch detected for point ({x}, {y}) on image {img_width}x{img_height}")
        return (int(x), int(y))

    for detection in detections:
        # Draw bounding box if available
        if "bbox" in detection:
            bbox = convert_bbox_to_pixels(detection["bbox"])
            draw.rectangle(bbox, outline="red", width=3)

            # Add label if available
            if "label" in detection:
                draw.text((bbox[0], bbox[1] - 10), detection["label"], fill="red")

        # Draw point if available
        elif "coordinates" in detection:
            coords = detection["coordinates"]
            if isinstance(coords, dict) and "x" in coords and "y" in coords:
                x, y = convert_point_to_pixels(coords["x"], coords["y"])
                # Draw crosshair
                draw.line([(x - 10, y), (x + 10, y)], fill="red", width=2)
                draw.line([(x, y - 10), (x, y + 10)], fill="red", width=2)

    # Create parent directory if needed (with error handling)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Cannot create directory: {output_path.parent}")
        if e.errno == errno.ENOSPC:
            raise IOError(f"Disk full: Cannot create directory {output_path.parent}")
        raise OSError(f"Failed to create output directory: {output_path.parent}\nError: {str(e)}")

    # Save annotated image
    try:
        annotated.save(output_path)
        logger.debug(f"Saved annotated image to: {output_path}")
    except OSError as e:
        logger.error(f"Cannot save image: {output_path}")
        if e.errno == errno.ENOSPC:
            raise IOError(f"Disk full: Cannot save image to {output_path}")
        raise OSError(f"Failed to save annotated image: {output_path}\nError: {str(e)}")

    return output_path
