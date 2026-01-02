import cv2
import numpy as np


class Visualizer:
    def __init__(self, config):
        self.config = config
        # Define standard colors (B, G, R)
        self.COLOR_PARKED = (0, 255, 0)  # Green
        self.COLOR_DRIVING = (0, 0, 255)  # Red
        self.COLOR_ZONE = (255, 191, 0)  # Deep Sky Blue
        self.COLOR_TEXT = (255, 255, 255)  # White
        self.COLOR_BG = (0, 0, 0)  # Black

    def draw_zone(self, frame, zone_coords):
        """Draws the parking corridor polygon."""
        pts = np.array(zone_coords, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=self.COLOR_ZONE, thickness=2)

        # Optional: Add a semi-transparent overlay to the zone
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], self.COLOR_ZONE)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
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

    def draw_dashboard(self, frame, total_parked, fps=None):
        """Draws the top status bar."""
        # Create a top banner
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

        # Text 2: Parked Count
        count_text = f"PARKED: {total_parked}"
        cv2.putText(
            frame,
            count_text,
            (w - 250, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.COLOR_PARKED,
            2,
        )

        return frame
