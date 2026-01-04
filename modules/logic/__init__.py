"""
Parking Occupancy Logic Module

This module contains the core logic for determining parking zone occupancy.
"""

from .zone_manager import ZoneManager
from .parking_zone import ParkingZone
from .geometry import (
    get_bottom_center,
    calculate_iou_with_zone,
    is_point_in_polygon,
    hybrid_detection
)

__all__ = [
    'ZoneManager',
    'ParkingZone',
    'get_bottom_center',
    'calculate_iou_with_zone',
    'is_point_in_polygon',
    'hybrid_detection'
]
