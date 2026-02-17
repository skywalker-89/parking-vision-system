import cv2
import numpy as np

class PerspectiveManager:
    def __init__(self, width=1920, height=1080):
        # SOURCE POINTS: Select 4 points from your camera image that form a 
        # perfect rectangle on the real ground (e.g., the road area).
        # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        # NOTE: You must tweak these coordinates to match your specific camera view!
        self.src_points = np.float32([
            [778, 42],    # Top-Left
            [1139, 34],   # Top-Right
            [1398, 913],  # Bottom-Right
            [537, 926]    # Bottom-Left
        ])

        # DESTINATION POINTS: Where we want them to go in the "Bird's Eye View"
        # We define a flat rectangle (e.g., 400x800 pixels)
        self.bev_w, self.bev_h = 400, 800
        self.dst_points = np.float32([
            [0, 0],              # Top-Left
            [self.bev_w, 0],     # Top-Right
            [self.bev_w, self.bev_h], # Bottom-Right
            [0, self.bev_h]      # Bottom-Left
        ])

        # Calculate the Matrix
        self.matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def transform_point(self, x, y):
        """
        Converts a point (x, y) from Camera View -> Bird's Eye View
        """
        point = np.array([[[x, y]]], dtype='float32')
        transformed = cv2.perspectiveTransform(point, self.matrix)
        return transformed[0][0] # Returns (x, y)

    def get_car_footprint(self, bbox):
        """
        Calculates the point where the car touches the ground (Bottom-Center).
        This is crucial! The top of the car doesn't matter, only the tires.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = y2  # The bottom of the box is where the tires are
        return self.transform_point(cx, cy)
    
    def get_spot_center_bev(self, spot_bbox):
        """
        Converts a parking spot center to BEV.
        """
        x1, y1, x2, y2 = spot_bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return self.transform_point(cx, cy)
