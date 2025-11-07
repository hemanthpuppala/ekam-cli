"""VLM response parsing utilities for detection and pointing endpoints."""

import json
import re
from typing import Any, Optional

from loguru import logger


def parse_detection_response(raw_response: str, object_name: str) -> list[dict]:
    """Parse VLM detection response to extract bounding boxes.

    Handles multiple formats:
    - JSON: [{"bbox": [x1,y1,x2,y2], "label": "car"}]
    - JSON with variations: bbox_2d, bounding_box, box, coordinates
    - Text with numbers: "car at [100, 200, 300, 400]"
    - Multiple detections in JSON array

    Args:
        raw_response: Raw text response from VLM
        object_name: Object that was being detected

    Returns:
        List of detection dicts: [{"label": str, "bbox": [x1,y1,x2,y2], "confidence": float}]
        Returns empty list if no valid detections found
    """
    detections = []

    # Try parsing as JSON first (most structured format)
    json_detections = _try_parse_json_detections(raw_response, object_name)
    if json_detections:
        return json_detections

    # Try extracting bounding boxes from text using regex
    text_detections = _try_parse_text_detections(raw_response, object_name)
    if text_detections:
        return text_detections

    # If no detections found, return empty list
    logger.warning(f"Could not parse detections from response: {raw_response[:200]}...")
    return []


def _try_parse_json_detections(raw_response: str, object_name: str) -> list[dict]:
    """Try to parse JSON-formatted detection response.

    Args:
        raw_response: Raw response text
        object_name: Object being detected

    Returns:
        List of parsed detections or empty list
    """
    logger.debug(f"_try_parse_json_detections called with object_name={object_name}")
    logger.debug(f"Raw response type: {type(raw_response)}, length: {len(str(raw_response))}")
    logger.debug(f"Raw response (FULL, no truncation): {raw_response}")

    try:
        # Extract JSON from markdown code blocks if present (try both array and object)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', raw_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            logger.debug(f"Found JSON in markdown code block")
        else:
            # Try to find JSON object (new format) first
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.debug(f"Found JSON object in text")
            else:
                # Try to find JSON array (old format)
                json_match = re.search(r'\[.*?\]', raw_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    logger.debug(f"Found JSON array in text")
                else:
                    logger.debug(f"No JSON found in response")
                    return []

        # Parse JSON
        logger.debug(f"Attempting to parse JSON: {json_str}")
        data = json.loads(json_str)

        # Handle new dict format: {"object_1": [x1, y1, x2, y2], "object_2": [x1, y1, x2, y2]}
        if isinstance(data, dict) and not any(k in data for k in ['bbox', 'bbox_2d', 'bounding_box', 'box', 'coordinates', 'coord', 'x_min', 'y_min', 'x_max', 'y_max']):
            logger.debug(f"Detected new dict format with keys: {list(data.keys())}")
            detections = []
            for key, bbox in data.items():
                if isinstance(bbox, list) and len(bbox) == 4:
                    # Extract base object name (remove _1, _2, etc.)
                    label = re.sub(r'_\d+$', '', key)
                    detection_dict = {
                        'label': label,
                        'bbox': bbox,
                        'confidence': 0.9
                    }
                    logger.debug(f"Adding detection from new format: {detection_dict}")
                    detections.append(detection_dict)

            if detections:
                logger.info(f"✓ Successfully parsed {len(detections)} detections from new dict format")
                return detections
            else:
                logger.warning(f"No valid detections in new dict format")
                return []

        # Old format: convert to list
        if not isinstance(data, list):
            data = [data]

        detections = []
        logger.debug(f"Processing {len(data)} items from parsed JSON")

        for idx, item in enumerate(data):
            logger.debug(f"Processing item {idx + 1}/{len(data)}: {item}")

            if not isinstance(item, dict):
                logger.debug(f"Item {idx + 1} is not a dict, skipping")
                continue

            # Extract bbox with various possible keys
            bbox = None

            # First check for array-style bbox
            for bbox_key in ['bbox', 'bbox_2d', 'bounding_box', 'box', 'coordinates', 'coord']:
                if bbox_key in item:
                    bbox = item[bbox_key]
                    logger.debug(f"Found bbox with key '{bbox_key}': {bbox}")
                    break

            # If not found, check for moondream-style x_min/y_min/x_max/y_max format
            is_moondream_format = False
            if not bbox and 'x_min' in item and 'y_min' in item and 'x_max' in item and 'y_max' in item:
                bbox = [
                    item['x_min'],
                    item['y_min'],
                    item['x_max'],
                    item['y_max']
                ]
                is_moondream_format = True
                logger.info(f"✓ Converted moondream bbox format to standard: {bbox}")

            if not bbox or len(bbox) != 4:
                logger.warning(f"Skipping detection with invalid bbox: {item}")
                continue

            # Extract label
            # Moondream doesn't return labels, so use object_name for moondream format
            if is_moondream_format and 'label' not in item:
                label = object_name
                logger.debug(f"Using object_name as label for moondream format: {label}")
            else:
                label = item.get('label', item.get('class', item.get('name', object_name)))
                logger.debug(f"Extracted label: {label}")
            # Remove quotes if present
            if isinstance(label, str):
                label = label.strip("'\"")

            # Extract confidence
            confidence = item.get('confidence', item.get('score', item.get('prob', 0.9)))

            detection_dict = {
                'label': label,
                'bbox': bbox,
                'confidence': float(confidence)
            }
            logger.debug(f"Adding detection {len(detections) + 1}: {detection_dict}")
            detections.append(detection_dict)

        if detections:
            logger.info(f"✓ Successfully parsed {len(detections)} detections from JSON response")
            logger.debug(f"Final detections list (FULL): {detections}")
        else:
            logger.warning(f"No detections parsed from JSON data")

        return detections

    except (json.JSONDecodeError, ValueError) as e:
        logger.debug(f"JSON parsing failed: {e}")
        return []


def _try_parse_text_detections(raw_response: str, object_name: str) -> list[dict]:
    """Try to extract bounding boxes from plain text response.

    Args:
        raw_response: Raw response text
        object_name: Object being detected

    Returns:
        List of parsed detections or empty list
    """
    detections = []

    # Pattern to match 4 numbers (bbox coordinates)
    # Matches: [100, 200, 300, 400] or (100, 200, 300, 400) or 100, 200, 300, 400
    bbox_pattern = r'[\[\(]?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)[\]\)]?'

    matches = re.finditer(bbox_pattern, raw_response)

    for match in matches:
        try:
            x1 = float(match.group(1))
            y1 = float(match.group(2))
            x2 = float(match.group(3))
            y2 = float(match.group(4))

            # Validate coordinates (must be positive and x2 > x1, y2 > y1)
            if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
                detections.append({
                    'label': object_name,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 0.9  # Default confidence for text-parsed detections
                })
        except ValueError:
            continue

    if detections:
        logger.info(f"Parsed {len(detections)} detections from text response")

    return detections


def parse_point_response(raw_response: str, object_name: str) -> dict:
    """Parse VLM pointing response to extract coordinates.

    Handles multiple formats:
    - JSON: {"x": 100, "y": 200}
    - Text: "coordinates: (100, 200)" or "at [100, 200]"
    - Text: "x=100, y=200"

    Args:
        raw_response: Raw text response from VLM
        object_name: Object that was being located

    Returns:
        Dict with x, y coordinates: {"x": int, "y": int}
        Returns center point {x: 50, y: 50} if parsing fails
    """
    # Try parsing as JSON first
    json_point = _try_parse_json_point(raw_response)
    if json_point:
        return json_point

    # Try extracting coordinates from text
    text_point = _try_parse_text_point(raw_response)
    if text_point:
        return text_point

    # Default fallback
    logger.warning(f"Could not parse point from response: {raw_response[:200]}...")
    return {"x": 50, "y": 50}


def _try_parse_json_point(raw_response: str) -> Optional[dict]:
    """Try to parse JSON-formatted point response.

    Args:
        raw_response: Raw response text

    Returns:
        Dict with x, y or None
    """
    try:
        # Extract JSON from markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object in the text
            json_match = re.search(r'\{.*?\}', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return None

        # Parse JSON
        data = json.loads(json_str)

        # Check for new format: {"object_name": [x, y]}
        # Look for any key that has an array value with 2 elements
        for key, value in data.items():
            if isinstance(value, list) and len(value) == 2:
                try:
                    x = float(value[0])
                    y = float(value[1])
                    logger.info(f"Parsed point from new dict format: ({x}, {y})")
                    return {"x": x, "y": y}
                except (ValueError, TypeError):
                    continue

        # Old format: Extract x, y with various possible keys
        x = None
        y = None

        for x_key in ['x', 'X', 'center_x', 'cx', 'coord_x']:
            if x_key in data:
                x = data[x_key]
                break

        for y_key in ['y', 'Y', 'center_y', 'cy', 'coord_y']:
            if y_key in data:
                y = data[y_key]
                break

        if x is not None and y is not None:
            logger.info(f"Parsed point from JSON: ({x}, {y})")
            return {"x": float(x), "y": float(y)}

        return None

    except (json.JSONDecodeError, ValueError) as e:
        logger.debug(f"JSON point parsing failed: {e}")
        return None


def _try_parse_text_point(raw_response: str) -> Optional[dict]:
    """Try to extract point coordinates from plain text response.

    Args:
        raw_response: Raw response text

    Returns:
        Dict with x, y or None
    """
    # Pattern to match coordinate pairs
    # Matches: (100, 200) or [100, 200] or x=100, y=200 or x:100, y:200

    # Try parentheses/brackets format first
    coord_pattern = r'[\[\(]\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*[\]\)]'
    match = re.search(coord_pattern, raw_response)

    if match:
        try:
            x = float(match.group(1))
            y = float(match.group(2))
            logger.info(f"Parsed point from text brackets: ({x}, {y})")
            return {"x": x, "y": y}
        except ValueError:
            pass

    # Try x=..., y=... format
    x_pattern = r'x\s*[:=]\s*(\d+\.?\d*)'
    y_pattern = r'y\s*[:=]\s*(\d+\.?\d*)'

    x_match = re.search(x_pattern, raw_response, re.IGNORECASE)
    y_match = re.search(y_pattern, raw_response, re.IGNORECASE)

    if x_match and y_match:
        try:
            x = float(x_match.group(1))
            y = float(y_match.group(1))
            logger.info(f"Parsed point from text key-value: ({x}, {y})")
            return {"x": x, "y": y}
        except ValueError:
            pass

    return None
