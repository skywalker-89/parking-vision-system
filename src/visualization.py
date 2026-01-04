import cv2
import numpy as np


class Visualizer:
    def __init__(self, config):
        self.config = config
        # Define standard colors (B, G, R)
        self.COLOR_PARKED = (0, 255, 0)  # Green
        self.COLOR_DRIVING = (0, 0, 255)  # Red
        self.COLOR_ZONE_OCCUPIED = (0, 200, 0)  # Green for occupied zones
        self.COLOR_ZONE_VACANT = (0, 0, 200)  # Red for vacant zones
        self.COLOR_TEXT = (255, 255, 255)  # White
        self.COLOR_BG = (0, 0, 0)  # Black

    def draw_zones(self, frame, zones):
        """
        Draw all parking zones on the frame.
        
        Args:
            frame: Video frame
            zones: Dict of {zone_id: ParkingZone objects}
        """
        for zone_id, zone in zones.items():
            # Get zone coordinates
            coords = list(zone.polygon.exterior.coords[:-1])  # Remove duplicate last point
            pts = np.array(coords, np.int32).reshape((-1, 1, 2))
            
            # Choose color based on occupancy
            status = zone.get_status()
            color = self.COLOR_ZONE_OCCUPIED if status == "OCCUPIED" else self.COLOR_ZONE_VACANT
            
            # Draw polygon outline
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=3)
            
            # Draw semi-transparent fill
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            
            # Draw zone label
            # Calculate center of polygon for label placement
            moments = cv2.moments(pts)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                
                # Create label with zone ID and status
                label = f"{zone_id}"
                status_short = "OCC" if status == "OCCUPIED" else "VAC"
                
                # Draw background for label
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (cx - w//2 - 5, cy - h - 5), 
                             (cx + w//2 + 5, cy + 5), self.COLOR_BG, -1)
                
                # Draw label text
                cv2.putText(frame, label, (cx - w//2, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw status below
                (w2, h2), _ = cv2.getTextSize(status_short, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.putText(frame, status_short, (cx - w2//2, cy + h + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame

    def draw_cars(self, frame, cars):
        """Draws bounding boxes and labels for all cars."""
        for car in cars:
            x1, y1, x2, y2 = map(int, car["bbox"])
            track_id = car["id"]
            is_parked = car.get("is_parked", False)  # Default to False if missing

            # 1. Pick Color
            color = self.COLOR_PARKED if is_parked else self.COLOR_DRIVING
            status_text = "PARKED" if is_parked else "DRIVING"

            # 2. Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 3. Draw Label Background (so text is readable)
            label = f"#{track_id} {status_text}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)

            # 4. Draw Text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.COLOR_TEXT,
                2,
            )

            # 5. Draw Anchor Point (Yellow Dot)
            if "center_point" in car:
                cx, cy = car["center_point"]
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        return frame

    def draw_dashboard(self, frame, total_occupied, total_zones=None):
        """
        Draws the top status bar with multi-zone information.
        
        Args:
            frame: Video frame
            total_occupied: Number of occupied zones
            total_zones: Total number of zones (optional)
        """
        h, w, _ = frame.shape
        # Draw a black strip at the top
        cv2.rectangle(frame, (0, 0), (w, 50), self.COLOR_BG, -1)

        # Text 1: Title
        cv2.putText(
            frame,
            "PARKING VISION SYSTEM",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.COLOR_TEXT,
            2,
        )

        # Text 2: Occupancy Count
        if total_zones:
            count_text = f"OCCUPIED: {total_occupied}/{total_zones}"
        else:
            count_text = f"OCCUPIED: {total_occupied}"
            
        cv2.putText(
            frame,
            count_text,
            (w - 300, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.COLOR_PARKED,
            2,
        )

        return frame
