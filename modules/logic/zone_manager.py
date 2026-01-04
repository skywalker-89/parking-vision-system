"""
ZoneManager class - Orchestrates multiple parking zones.

Responsibilities:
- Load and manage multiple parking zones
- Assign vehicles to appropriate zones
- Coordinate updates across all zones
- Provide aggregated occupancy data
"""

from .parking_zone import ParkingZone


class ZoneManager:
    """
    Manages multiple parking zones and their occupancy states.
    """
    
    def __init__(self, config):
        """
        Initialize the zone manager with configuration.
        
        Args:
            config: Dict containing full configuration
                - zones: List of zone configurations
                - detection: Detection threshold settings
        """
        self.config = config
        self.zones = {}
        
        detection_config = config.get('detection', {})
        
        # Create ParkingZone objects for each configured zone
        for zone_config in config.get('zones', []):
            zone_id = zone_config['id']
            self.zones[zone_id] = ParkingZone(zone_config, detection_config)
    
    def update(self, detected_vehicles):
        """
        Update all zones with currently detected vehicles.
        
        Args:
            detected_vehicles: List of vehicle dicts from detector
                Each dict should have: id, bbox, center_point
        """
        # Update each zone independently
        for zone in self.zones.values():
            zone.update(detected_vehicles)
    
    def get_occupancy_summary(self):
        """
        Get summary of occupancy status for all zones.
        
        Returns:
            Dict mapping zone_id to status information:
            {
                "SLOT_A1": {
                    "status": "OCCUPIED",
                    "vehicle_ids": [42],
                    "count": 1,
                    "capacity": 1
                },
                ...
            }
        """
        summary = {}
        for zone_id, zone in self.zones.items():
            summary[zone_id] = {
                'status': zone.get_status(),
                'vehicle_ids': zone.get_parked_vehicles(),
                'count': zone.get_count(),
                'capacity': zone.capacity,
                'over_capacity': zone.is_over_capacity()
            }
        return summary
    
    def get_total_occupied(self):
        """
        Get total number of occupied zones.
        
        Returns:
            Integer count of occupied zones
        """
        return sum(1 for zone in self.zones.values() if zone.get_status() == "OCCUPIED")
    
    def get_total_vacant(self):
        """
        Get total number of vacant zones.
        
        Returns:
            Integer count of vacant zones
        """
        return sum(1 for zone in self.zones.values() if zone.get_status() == "VACANT")
    
    def get_total_capacity(self):
        """
        Get total parking capacity across all zones.
        
        Returns:
            Integer total capacity
        """
        return sum(zone.capacity for zone in self.zones.values())
    
    def get_zone_by_id(self, zone_id):
        """
        Get a specific zone by its ID.
        
        Args:
            zone_id: String identifier of the zone
            
        Returns:
            ParkingZone object or None if not found
        """
        return self.zones.get(zone_id)
    
    def get_all_zones(self):
        """
        Get all zones as a dict.
        
        Returns:
            Dict mapping zone_id to ParkingZone objects
        """
        return self.zones
