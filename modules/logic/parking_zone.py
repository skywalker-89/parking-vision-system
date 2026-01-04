"""
ParkingZone class - Represents a single parking slot with occupancy tracking.

Handles:
- Temporal logic (entry/exit thresholds)
- Vehicle tracking within the zone
- Status determination (OCCUPIED/VACANT)
"""

from shapely.geometry import Polygon
from .geometry import hybrid_detection


class ParkingZone:
    """
    Represents a single parking zone with independent occupancy tracking.
    """
    
    def __init__(self, zone_config, detection_config):
        """
        Initialize a parking zone.
        
        Args:
            zone_config: Dict containing zone configuration
                - id: Unique identifier (e.g., "SLOT_A1")
                - polygon: List of [x, y] coordinates
                - capacity: Number of vehicles that can park here
                - type: Zone type (e.g., "parallel", "angled")
            detection_config: Dict containing detection thresholds
                - entry_threshold_frames: Frames needed to confirm parking
                - exit_threshold_frames: Frames needed to confirm departure
                - overlap_ratio_threshold: IoU threshold for overlap detection
        """
        self.id = zone_config['id']
        self.polygon = Polygon(zone_config['polygon'])
        self.capacity = zone_config.get('capacity', 1)
        self.zone_type = zone_config.get('type', 'unknown')
        
        # Detection thresholds
        self.entry_threshold = detection_config.get('entry_threshold_frames', 30)
        self.exit_threshold = detection_config.get('exit_threshold_frames', 90)
        self.overlap_threshold = detection_config.get('overlap_ratio_threshold', 0.2)
        
        # Tracking state
        self.vehicle_history = {}  # {vehicle_id: tracking_data}
        self.parked_vehicles = set()  # Set of vehicle IDs currently parked
    
    def update(self, detected_vehicles):
        """
        Update zone state with currently detected vehicles.
        
        Args:
            detected_vehicles: List of dicts with keys: id, bbox, center_point
        """
        # Track which vehicles are currently in this zone
        vehicles_in_zone = set()
        
        for vehicle in detected_vehicles:
            vehicle_id = vehicle['id']
            bbox = vehicle['bbox']
            
            # Check if vehicle is in this zone
            is_inside = hybrid_detection(bbox, self.polygon, self.overlap_threshold)
            
            # DEBUG: Print detection details for troubleshooting
            # Uncomment the next 3 lines to see why vehicles aren't detected
            # from .geometry import get_bottom_center, calculate_iou_with_zone
            # anchor = get_bottom_center(bbox)
            # print(f"  Zone {self.id} | Car #{vehicle_id} | Anchor: {anchor} | Inside: {is_inside}")
            
            # Initialize tracking if new vehicle
            if vehicle_id not in self.vehicle_history:
                self.vehicle_history[vehicle_id] = {
                    'frames_inside': 0,
                    'frames_outside': 0,
                    'status': 'DRIVING'
                }
            
            history = self.vehicle_history[vehicle_id]
            
            # Update temporal counters
            if is_inside:
                vehicles_in_zone.add(vehicle_id)
                history['frames_inside'] += 1
                history['frames_outside'] = 0
                
                # Entry logic: Mark as parked if inside long enough
                if history['frames_inside'] >= self.entry_threshold:
                    self.parked_vehicles.add(vehicle_id)
                    history['status'] = 'PARKED'
            else:
                history['frames_inside'] = 0
                history['frames_outside'] += 1
        
        # Exit logic: Remove vehicles that have been gone long enough
        vehicles_to_remove = set()
        for vehicle_id in self.parked_vehicles:
            if vehicle_id not in vehicles_in_zone:
                history = self.vehicle_history.get(vehicle_id, {})
                if history.get('frames_outside', 0) >= self.exit_threshold:
                    vehicles_to_remove.add(vehicle_id)
                    history['status'] = 'DRIVING'
        
        self.parked_vehicles -= vehicles_to_remove
    
    def get_status(self):
        """
        Get current occupancy status of this zone.
        
        Returns:
            String: "OCCUPIED" or "VACANT"
        """
        return "OCCUPIED" if len(self.parked_vehicles) > 0 else "VACANT"
    
    def get_parked_vehicles(self):
        """
        Get list of vehicle IDs currently parked in this zone.
        
        Returns:
            List of vehicle IDs
        """
        return list(self.parked_vehicles)
    
    def get_count(self):
        """
        Get number of vehicles currently parked in this zone.
        
        Returns:
            Integer count
        """
        return len(self.parked_vehicles)
    
    def is_over_capacity(self):
        """
        Check if zone is over its designated capacity.
        
        Returns:
            Boolean
        """
        return len(self.parked_vehicles) > self.capacity
