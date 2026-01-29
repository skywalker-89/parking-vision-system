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

        for spot in spot_statuses:
            x1, y1, x2, y2 = spot["bbox"]
            occupied = spot["occupied"]

            color = self.COLOR_OCCUPIED if occupied else self.COLOR_FREE

            # 1. Draw the Spot Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 2. Draw Overlay (Transparent Fill) if occupied
            if occupied:
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

                # Show which car is taking the spot
                if spot["car_id"]:
                    label = f"Taken: #{spot['car_id']}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        self.COLOR_TEXT,
                        1,
                    )

        return frame

    def draw_cars(self, frame, cars):
        """Draws cars (Simplified since spots now handle the main colors)."""
        for car in cars:
            x1, y1, x2, y2 = map(int, car["bbox"])

            # We assume the spot visualizer handles the red/green logic.
            # Here we just draw a subtle box for the car so we know it's detected.
            color = (255, 200, 0)  # Orange/Yellow for raw car detection

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            # Draw Center Point (Crucial for debugging alignment)
            if "center_point" in car:
                cx, cy = car["center_point"]
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

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
            "AI PARKING SYSTEM",
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
