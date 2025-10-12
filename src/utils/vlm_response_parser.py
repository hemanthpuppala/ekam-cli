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
    try:
        # Extract JSON from markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', raw_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON array in the text
            json_match = re.search(r'\[.*?\]', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return []

        # Parse JSON
        data = json.loads(json_str)

        if not isinstance(data, list):
            data = [data]

        detections = []
        for item in data:
            if not isinstance(item, dict):
                continue

            # Extract bbox with various possible keys
            bbox = None
            for bbox_key in ['bbox', 'bbox_2d', 'bounding_box', 'box', 'coordinates', 'coord']:
                if bbox_key in item:
                    bbox = item[bbox_key]
                    break

            if not bbox or len(bbox) != 4:
                logger.debug(f"Skipping detection with invalid bbox: {item}")
                continue

            # Extract label
            label = item.get('label', item.get('class', item.get('name', object_name)))
            # Remove quotes if present
            if isinstance(label, str):
                label = label.strip("'\"")

            # Extract confidence
            confidence = item.get('confidence', item.get('score', item.get('prob', 0.9)))

            detections.append({
                'label': label,
                'bbox': bbox,
                'confidence': float(confidence)
            })

        if detections:
            logger.info(f"Parsed {len(detections)} detections from JSON response")

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

        # Extract x, y with various possible keys
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
