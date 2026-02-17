# Parking Vision System

A robust, AI-powered system for detecting parking spot occupancy using computer vision. This project utilizes a **Dual-Model Architecture** combined with **Bird's Eye View (BEV)** logic to accurately monitor parking availability, even under challenging conditions.

---

## 🚀 Features
1.  **Dual-Model Architecture**:
    *   **Spot Detection (`spots.pt`)**: Automatically identifies parking spot lines and locations (runs once or on trigger).
    *   **Car Detection (`best.pt`)**: Real-time YOLOv8 model to detect cars in every frame.
2.  **Smart Occupancy Logic**: Uses Perspective Transformation (Homography) to map cars to spots in a "Bird's Eye View", solving perspective distortion issues.
3.  **Dynamic Updates**: Includes a simulation clock (24h cycle). At **2 AM**, if traffic is low, it re-calibrates spot positions to account for camera movement.
4.  **Robust Visualization**: Clear overlays, occupancy dashboard, and simulation time display.

---

## 🛠️ Installation

### Prerequisites
*   Python 3.8 or higher
*   `pip` (Python Package Installer)

### Setup
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/parking-vision-system.git
    cd parking-vision-system
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃 Usage

### 1. Configuration
The system is pre-configured, but you can adjust settings in `config/config.yaml` if needed.

### 2. Run the System
Execute the main script from the root directory:

```bash
python3 src/main.py
```

### 3. Controls
*   **press 'q'**: Quit the application.

---

## 🧠 How It Works (Project Flow)

The system follows a strict pipeline to ensure accuracy:

### Phase 1: Initialization
Before processing the video, the system "learns" the parking lot layout:
1.  **Check for Saved Data**: Looks for `config/spots_data.json`.
2.  **Spot Detection (If no data)**:
    *   It loads a reference image (`config/reference.jpg`).
    *   Runs the **Spot Model (`spots.pt`)** to find empty spots.
    *   Saves these coordinates to JSON for future use.
3.  **Fallback**: If all else fails, it generates a manual demo grid.

### Phase 2: The Main Loop (Frame-by-Frame)
For every frame of the video:

1.  **Car Detection**:
    *   The **Car Model (`best.pt`)** scans the entire frame.
    *   It returns bounding boxes for all detected vehicles.

2.  **Occupancy Matching (The "Brain")**:
    *   **Perspective Transform**: The system converts the *bottom-center* of every car (tires) and the *center* of every spot into **Bird's Eye View (BEV)** coordinates.
    *   **Distance Check**: It calculates the distance between cars and spots in this flat 2D plane.
    *   **Decision**: If a car is within a threshold distance of a spot, that spot is marked `OCCUPIED`.

3.  **Simulation & Maintenance**:
    *   The system tracks a simulated "Time of Day".
    *   **2 AM Self-Healing**: If it's 2 AM and the lot is mostly empty, the system triggers a re-scan of the parking lines (`spots.pt`) to fix any drift.

### Phase 3: Visualization
Finally, the system draws the results:
*   **Green Box**: Free Spot.
*   **Red Box**: Occupied Spot.
*   **Dashboard**: Shows real-time counts at the top of the screen.

---

## 📂 Project Structure

```
parking-vision-system/
├── config/                 # Configuration files and reference images
│   ├── config.yaml         # General settings
│   ├── spots_data.json     # Saved spot coordinates
│   └── reference.jpg       # Reference image for spot detection
├── modules/
│   ├── detector/           # Car detection logic (best.pt)
│   └── parking_logic/      # Spot detection (spots.pt) & occupancy logic
├── src/
│   ├── main.py             # Entry point of the application
│   └── visualization.py    # Drawing utilities
├── videos/                 # Input video files
└── requirements.txt        # Python dependencies
```

---

## 🔧 Troubleshooting

*   **"Model not found"**: Ensure `best.pt` is in `modules/detector/` and `spots.pt` is in `modules/parking_logic/`.
*   **"Video not found"**: Check the `VIDEO_PATH` in `src/main.py` matches your video filename in `videos/`.
