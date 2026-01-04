"""
Geometric utility functions for parking zone detection.

Provides functions for:
- Calculating anchor points (where vehicle touches ground)
- Point-in-polygon detection
- Intersection over Union (IoU) calculations
- Hybrid detection combining multiple methods
"""

from shapely.geometry import Point, Polygon


def get_bottom_center(bbox):
    """
    Calculate the bottom-center point of a bounding box.
    This represents where the vehicle's wheels touch the ground.
    
    Args:
        bbox: List or tuple of [x1, y1, x2, y2] coordinates
        
    Returns:
        Tuple of (cx, cy) representing the anchor point
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) // 2)  # Horizontal center
    cy = int(y2)               # Bottom edge
    return (cx, cy)


def calculate_iou_with_zone(bbox, zone_polygon):
    """
    Calculate how much of the parking zone is covered by the vehicle bounding box.
    
    Args:
        bbox: List or tuple of [x1, y1, x2, y2] coordinates
        zone_polygon: Shapely Polygon object representing the parking zone
        
    Returns:
        Float between 0.0 and 1.0 representing overlap ratio
        (intersection area / zone area)
    """
    x1, y1, x2, y2 = bbox
    bbox_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    
    try:
        intersection_area = zone_polygon.intersection(bbox_polygon).area
        zone_area = zone_polygon.area
        overlap_ratio = intersection_area / zone_area if zone_area > 0 else 0.0
        return overlap_ratio
    except Exception:
        # Handle any geometric edge cases
        return 0.0


def is_point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon.
    
    Args:
        point: Tuple of (x, y) coordinates
        polygon: Shapely Polygon object
        
    Returns:
        Boolean indicating if point is inside polygon
    """
    point_obj = Point(point)
    return polygon.contains(point_obj)


def hybrid_detection(bbox, zone_polygon, overlap_threshold=0.2):
    """
    Determine if a vehicle is in a zone using hybrid detection.
    Combines point-in-polygon (primary) with IoU overlap (fallback).
    
    Args:
        bbox: List or tuple of [x1, y1, x2, y2] coordinates
        zone_polygon: Shapely Polygon object representing the parking zone
        overlap_threshold: Minimum overlap ratio to consider vehicle in zone (default: 0.2)
        
    Returns:
        Boolean indicating if vehicle is considered to be in the zone
    """
    # Method 1: Check if anchor point is inside zone
    anchor_point = get_bottom_center(bbox)
    point_inside = is_point_in_polygon(anchor_point, zone_polygon)
    
    # Method 2: Check overlap ratio (for edge cases)
    overlap_ratio = calculate_iou_with_zone(bbox, zone_polygon)
    
    # Vehicle is in zone if EITHER condition is met
    return point_inside or (overlap_ratio > overlap_threshold)
