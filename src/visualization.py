import cv2
import numpy as np


class Visualizer:
    def __init__(self, config):
        self.config = config
        self.COLOR_FREE = (0, 255, 0)  # Green
        self.COLOR_OCCUPIED = (0, 0, 255)  # Red
        self.COLOR_TEXT = (255, 255, 255)  # White
        self.COLOR_BG = (0, 0, 0)  # Black

    def draw_spots(self, frame, spot_statuses):
        """Draws the grid of parking spots with status."""
        
        # Create a single overlay for all transparency operations
        overlay = frame.copy()
        
        for spot in spot_statuses:
            x1, y1, x2, y2 = spot["bbox"]
            occupied = spot["occupied"]
            
            # Determine Color/Text
            if occupied:
                color = self.COLOR_OCCUPIED
                text = "TAKEN"
            else:
                color = self.COLOR_FREE
                text = "FREE"

            # 1. Draw Fill on Overlay (Transparency layer)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # 2. Draw Text Label
            # Calculate centered text position
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            cv2.putText(
                frame, # Draw text on the sharp frame, not overlay
                text,
                (cx - text_w // 2, cy + text_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.COLOR_TEXT,
                1,
                cv2.LINE_AA
            )

        # 3. Apply the overlay blending ONCE
        alpha = 0.3 # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 4. Draw Borders (Sharp lines on top)
        for spot in spot_statuses:
            x1, y1, x2, y2 = spot["bbox"]
            occupied = spot["occupied"]
            color = self.COLOR_OCCUPIED if occupied else self.COLOR_FREE
            # Thicker border for better visibility
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        return frame

    def draw_cars(self, frame, cars):
        """
        Draws cars.
        For Non-Engineer view, we hide the raw car boxes to avoid clutter.
        Only the Spot status matters.
        """
        # UNCOMMENT below to debug raw detections
        # for car in cars:
        #     x1, y1, x2, y2 = map(int, car["bbox"])
        #     color = (255, 200, 0)
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        return frame

    def draw_dashboard(self, frame, total_spots, occupied_spots):
        """Draws the top status bar."""
        h, w, _ = frame.shape
        available = total_spots - occupied_spots

        # Background
        cv2.rectangle(frame, (0, 0), (w, 60), self.COLOR_BG, -1)

        # Title
        cv2.putText(
            frame,
            "NMJ PARKING SYSTEM",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.COLOR_TEXT,
            2,
        )

        # Stats
        stat_text = (
            f"FREE: {available}  |  OCCUPIED: {occupied_spots}  |  TOTAL: {total_spots}"
        )

        # Color based on availability
        stat_color = self.COLOR_FREE if available > 0 else self.COLOR_OCCUPIED

        # Right align
        (text_w, _), _ = cv2.getTextSize(stat_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(
            frame,
            stat_text,
            (w - text_w - 20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            stat_color,
            2,
        )

        return frame