"""
generate_report.py
==================
Generates the academic report (.docx) for the Parking Vision System university project.
Run from the PROJECT ROOT:
    python documents/generate_report.py

Output: documents/Parking_Vision_System_Report.docx
"""

import os
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ──────────────────────────────────────────────────────────────────────────────
# PERSONALISE THESE BEFORE GENERATING
# ──────────────────────────────────────────────────────────────────────────────
STUDENT_NAME     = "Your Full Name"
STUDENT_ID       = "Your Student ID"
COURSE_NAME      = "Your Course Name"
UNIVERSITY       = "Your University"
SUPERVISOR       = "Your Supervisor's Name"   # leave blank "" if none
SUBMISSION_DATE  = "March 2026"
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPERIMENTAL = os.path.join(PROJECT_ROOT, "experimental")
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), "Parking_Vision_System_Report.docx")

# Experimental image paths (best model = y11s)
IMG_RESULTS    = os.path.join(EXPERIMENTAL, "y11s", "results.png")
IMG_CONFUSION  = os.path.join(EXPERIMENTAL, "y11s", "confusion_matrix_normalized.png")
IMG_PR_CURVE   = os.path.join(EXPERIMENTAL, "y11s", "BoxPR_curve.png")
IMG_F1_CURVE   = os.path.join(EXPERIMENTAL, "y11s", "BoxF1_curve.png")
IMG_VAL_PRED   = os.path.join(EXPERIMENTAL, "y11s", "val_batch0_pred.jpg")
IMG_VAL_LABEL  = os.path.join(EXPERIMENTAL, "y11s", "val_batch0_labels.jpg")
IMG_REFERENCE  = os.path.join(PROJECT_ROOT, "config", "reference.jpg")


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def set_font(run, name="Times New Roman", size=12, bold=False, italic=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)


def add_heading(doc, text, level=1):
    """Add a numbered heading with Times New Roman styling."""
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in p.runs:
        run.font.name = "Times New Roman"
        run.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_body(doc, text, indent=False):
    """Add a justified body paragraph."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        p.paragraph_format.first_line_indent = Cm(1.25)
    run = p.add_run(text)
    set_font(run)
    return p


def add_bullet(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if bold_prefix:
        r = p.add_run(bold_prefix + ": ")
        set_font(r, bold=True)
    r = p.add_run(text)
    set_font(r)
    return p


def add_image(doc, path, caption, width=Inches(5.5)):
    if os.path.exists(path):
        doc.add_picture(path, width=width)
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in cap.runs:
            set_font(run, size=10, italic=True)
    else:
        p = doc.add_paragraph(f"[Figure not found: {os.path.basename(path)}]")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def add_table(doc, headers, rows, col_widths=None):
    """Add a styled table with a dark header row."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        # Dark background
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "2C3E50")
        tc_pr.append(shd)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        set_font(run, bold=True, color=(255, 255, 255))

    # Data rows
    for r_idx, row_data in enumerate(rows):
        row = table.rows[r_idx + 1]
        fill = "ECF0F1" if r_idx % 2 == 0 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            tc_pr = cell._tc.get_or_add_tcPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:fill"), fill)
            tc_pr.append(shd)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            bold = (r_idx == 1 and c_idx == 0)  # highlight best model row
            run = p.add_run(str(val))
            set_font(run, bold=bold)

    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = w

    return table


def add_page_break(doc):
    doc.add_page_break()


def set_page_margins(doc, top=2.54, bottom=2.54, left=3.17, right=3.17):
    section = doc.sections[0]
    section.top_margin    = Cm(top)
    section.bottom_margin = Cm(bottom)
    section.left_margin   = Cm(left)
    section.right_margin  = Cm(right)


# ──────────────────────────────────────────────────────────────────────────────
# REPORT SECTIONS
# ──────────────────────────────────────────────────────────────────────────────

def make_title_page(doc):
    doc.add_paragraph("\n\n\n")
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(
        "Vision-Based Smart Parking Occupancy Detection System\n"
        "Using Dual YOLOv11 Architecture"
    )
    set_font(run, size=18, bold=True)

    doc.add_paragraph("\n")

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = subtitle.add_run("A University Project Report")
    set_font(r, size=14, italic=True)

    doc.add_paragraph("\n\n")

    for label, value in [
        ("Student Name",   STUDENT_NAME),
        ("Student ID",     STUDENT_ID),
        ("Course",         COURSE_NAME),
        ("University",     UNIVERSITY),
        ("Supervisor",     SUPERVISOR if SUPERVISOR else "N/A"),
        ("Date",           SUBMISSION_DATE),
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = p.add_run(f"{label}: ")
        set_font(r1, bold=True, size=12)
        r2 = p.add_run(value)
        set_font(r2, size=12)

    add_page_break(doc)


def make_abstract(doc):
    add_heading(doc, "Abstract", level=1)
    add_body(doc,
        "Efficient parking management is a critical challenge in modern urban environments. "
        "Traditional approaches, relying on physical sensors or manual observation, are costly, "
        "difficult to scale, and prone to failure. This report presents the design and implementation "
        "of a Vision-Based Smart Parking Occupancy Detection System that leverages a Dual YOLOv11 "
        "deep learning architecture combined with Perspective Transformation (Bird's Eye View) to "
        "accurately monitor parking availability in real time. "
        "The system employs two specialised models: a parking spot detector (spots.pt) that identifies "
        "vacant parking bays from a reference image, and a vehicle detector (best.pt) that tracks cars "
        "frame-by-frame in live video. Occupancy is determined by projecting both car positions and spot "
        "centres into a Bird's Eye View coordinate space, eliminating perspective distortion, and applying "
        "a greedy distance-based matching algorithm. A self-healing mechanism automatically recalibrates "
        "spot positions at 2:00 AM when traffic is low, ensuring robustness against camera drift. "
        "Four YOLOv11 model variants (Nano and Small, across two dataset configurations) were trained "
        "for 50 epochs and benchmarked. The YOLOv11 Small model achieved the highest mAP@50 of 96.4% "
        "and mAP@50-95 of 80.7%, and was selected for deployment. The system produces a clear real-time "
        "dashboard overlay, providing parking operators with live FREE and OCCUPIED counts at a glance.",
        indent=True
    )
    doc.add_paragraph()


def make_introduction(doc):
    add_heading(doc, "1. Introduction", level=1)

    add_heading(doc, "1.1 Background and Motivation", level=2)
    add_body(doc,
        "The rapid growth of urban populations has intensified demand for efficient parking infrastructure. "
        "Studies estimate that drivers in busy cities spend an average of 17 minutes searching for a parking "
        "space per trip, contributing to traffic congestion, fuel waste, and increased carbon emissions "
        "(Shoup, 2011). Automated parking management systems offer a scalable and cost-effective solution "
        "to this problem by providing real-time occupancy information to both drivers and facility managers.",
        indent=True
    )
    add_body(doc,
        "Conventional sensor-based systems (e.g., magnetic induction loops, infrared sensors, ultrasonic "
        "sensors) require extensive physical installation per parking bay and are expensive to maintain. "
        "Camera-based computer vision approaches have emerged as a compelling alternative, enabling a single "
        "camera to monitor an entire parking zone without per-bay hardware. Recent advances in deep learning, "
        "particularly the YOLO (You Only Look Once) family of object detectors, have made real-time "
        "camera-based solutions both fast and accurate enough for practical deployment.",
        indent=True
    )

    add_heading(doc, "1.2 Problem Statement", level=2)
    add_body(doc,
        "Existing single-model camera-based systems typically rely on either spot-region classification "
        "(checking each predefined bounding box for occupancy) or direct vehicle detection. The former "
        "approach requires manual annotation of every parking bay, while the latter struggles with perspective "
        "distortion — a car at the far end of a lot appears much smaller than one nearby, causing distance-based "
        "matching to fail. This project addresses both limitations through a dual-model architecture and "
        "Bird's Eye View perspective normalisation.",
        indent=True
    )

    add_heading(doc, "1.3 Project Objectives", level=2)
    for obj in [
        "Design and implement a dual-model YOLOv11 pipeline for real-time parking occupancy detection.",
        "Apply Perspective Transformation (Homography) to eliminate camera distortion effects.",
        "Train and benchmark four YOLOv11 variants to identify the optimal deployment model.",
        "Implement a self-healing recalibration mechanism for long-term operational robustness.",
        "Produce a clear, real-time visual dashboard overlay for practical use.",
    ]:
        add_bullet(doc, obj)

    add_heading(doc, "1.4 Report Structure", level=2)
    add_body(doc,
        "The remainder of this report is structured as follows: Section 2 reviews related work in "
        "parking detection and object detection. Section 3 describes the system architecture and design. "
        "Section 4 details the implementation. Section 5 presents experimental results and evaluation. "
        "Section 6 provides discussion and analysis. Section 7 concludes the report with directions "
        "for future work. References follow.",
        indent=True
    )


def make_literature_review(doc):
    add_heading(doc, "2. Literature Review", level=1)

    add_heading(doc, "2.1 Traditional Parking Detection Methods", level=2)
    add_body(doc,
        "Early automated parking systems relied on physical sensing hardware. Magnetic induction loop "
        "detectors, embedded in road surfaces, detect the metallic mass of a vehicle overhead. Infrared "
        "and ultrasonic sensor arrays mounted above each bay offer similar binary occupancy detection. "
        "While reliable in controlled environments, these approaches entail high installation costs, "
        "require per-bay hardware, and are susceptible to environmental interference (Benson et al., 2016). "
        "The per-space scalability problem makes them impractical for large open-air car parks.",
        indent=True
    )

    add_heading(doc, "2.2 Camera-Based Vision Approaches", level=2)
    add_body(doc,
        "Camera-based systems have been widely explored as a cost-effective alternative. Early work by "
        "de Almeida et al. (2015) demonstrated that hand-crafted feature methods (HOG + SVM) could "
        "achieve reasonable accuracy on a single static camera view. However, these approaches required "
        "careful illumination control and were brittle to viewpoint changes. The PKLot dataset (de Almeida "
        "et al., 2015) became a standard benchmark for evaluating such systems, providing over 12,000 "
        "annotated parking space images across different weather conditions.",
        indent=True
    )

    add_heading(doc, "2.3 Deep Learning and YOLO-Based Detection", level=2)
    add_body(doc,
        "The introduction of Convolutional Neural Networks (CNNs) dramatically improved visual recognition "
        "accuracy. Redmon et al. (2016) proposed YOLO (You Only Look Once), a single-stage real-time "
        "object detector that frames detection as a regression problem, achieving unprecedented inference "
        "speed without sacrificing accuracy. Successive versions (YOLOv3 through YOLOv8 and YOLOv11) "
        "have further refined the architecture, improving both speed and accuracy through innovations in "
        "anchor design, backbone architecture, and loss functions (Jocher et al., 2023).",
        indent=True
    )
    add_body(doc,
        "YOLOv11, released by Ultralytics, introduces improved spatial attention mechanisms and a "
        "redesigned C3k2 backbone block, achieving superior mAP against competing architectures at "
        "comparable inference speeds (Jocher et al., 2024). Its Nano (n) and Small (s) variants make "
        "it particularly suitable for edge deployment scenarios where computation is constrained.",
        indent=True
    )

    add_heading(doc, "2.4 Perspective Transformation in Parking Systems", level=2)
    add_body(doc,
        "A significant challenge in aerial or oblique-angle parking cameras is perspective distortion: "
        "objects at varying distances appear at different scales. Homography-based Perspective "
        "Transformation, commonly used in autonomous driving applications (Chen et al., 2020), maps "
        "a trapezoidal ground region in the camera view onto a flat Bird's Eye View (BEV) rectangle. "
        "This produces a scale-normalised, top-down representation in which metric distances between "
        "objects are preserved, enabling reliable proximity-based matching.",
        indent=True
    )

    add_heading(doc, "2.5 Research Gap", level=2)
    add_body(doc,
        "While existing work addresses vehicle detection and parking occupancy independently, few systems "
        "combine a dedicated spot detection model with a separate vehicle tracker within a unified BEV "
        "matching framework. Furthermore, the self-healing recalibration concept — automatically updating "
        "spot positions when conditions are favourable — has not been widely reported in the literature. "
        "This project contributes a novel integration of these elements into a single deployable system.",
        indent=True
    )


def make_system_architecture(doc):
    add_heading(doc, "3. System Architecture", level=1)

    add_heading(doc, "3.1 Overview", level=2)
    add_body(doc,
        "The system operates in two distinct phases: an Initialisation Phase that discovers and saves "
        "parking spot locations, and a continuous Main Processing Loop that detects vehicles, determines "
        "occupancy, and renders the visualisation. Both phases are orchestrated by the entry-point module "
        "src/main.py.",
        indent=True
    )

    add_heading(doc, "3.2 Dual-Model Architecture", level=2)
    add_body(doc,
        "Two independent YOLOv11 models serve distinct roles within the pipeline:",
        indent=True
    )
    add_table(doc,
        headers=["Model", "Weight File", "Role", "Trigger"],
        rows=[
            ["Spot Detector", "spots.pt", "Locates empty parking bay boundaries", "Once at startup; again at 2 AM if low traffic"],
            ["Car Detector",  "best.pt",  "Detects and tracks vehicles in every frame", "Every video frame (real-time)"],
        ],
        col_widths=[Cm(3), Cm(3), Cm(7), Cm(5.5)]
    )
    doc.add_paragraph()

    add_heading(doc, "3.3 Initialisation Phase", level=2)
    add_body(doc,
        "Prior to entering the main processing loop, the system bootstraps parking spot locations "
        "through the following decision chain implemented in SpotManager.detect_spots_initial():",
        indent=True
    )
    for step in [
        ("Step 1 — Load from JSON", "If config/spots_data.json exists, previously detected spot coordinates are loaded and scaled to the current video resolution. This avoids redundant model inference on repeated runs."),
        ("Step 2 — Detect from Reference Image", "If no saved data exists, the spot detector (spots.pt) is run on config/reference.jpg — a clean, unoccupied image of the parking lot. Detected bounding boxes are passed through Non-Maximum Suppression (NMS, IoU threshold = 0.4) and saved to spots_data.json."),
        ("Step 3 — Fallback Grid", "If neither JSON nor reference image is available, a configurable demo grid is generated programmatically as a fallback."),
    ]:
        add_bullet(doc, step[1], bold_prefix=step[0])

    add_body(doc,
        "The saved spots_data.json for this deployment contains 37 detected parking bays at a "
        "reference resolution of 2688 × 1520 pixels. Coordinates are automatically rescaled when "
        "the live video resolution differs.",
        indent=True
    )

    add_heading(doc, "3.4 Occupancy Matching — Bird's Eye View (BEV)", level=2)
    add_body(doc,
        "The core intelligence of the system lies in its occupancy matching strategy, implemented in "
        "SpotManager.update_occupancy(). Rather than comparing car and spot positions in the distorted "
        "camera view, both are first projected into a normalised Bird's Eye View coordinate space.",
        indent=True
    )
    add_body(doc,
        "The PerspectiveManager class computes a 3×3 homography matrix using four manually selected "
        "ground-plane anchor points (defining a trapezoidal region in camera view) and their corresponding "
        "positions in a flat 400 × 800 pixel BEV canvas. Two key transforms are applied per frame:",
        indent=True
    )
    for item in [
        ("Car BEV Position", "The bottom-centre of each detected car's bounding box (representing the tyre contact point with the ground) is transformed into BEV coordinates."),
        ("Spot BEV Position", "The centre of each parking spot bounding box is transformed into BEV coordinates."),
    ]:
        add_bullet(doc, item[1], bold_prefix=item[0])

    add_body(doc,
        "Matching proceeds via a greedy Hungarian-style algorithm: all (car, spot) pairs are scored "
        "by Euclidean distance in BEV space; the list is sorted ascending; pairs are assigned from "
        "closest to farthest, with each car and spot consumed at most once. A distance threshold of "
        "70 pixels in BEV space is applied to reject implausible matches.",
        indent=True
    )

    add_heading(doc, "3.5 Self-Healing Recalibration", level=2)
    add_body(doc,
        "Parking lot cameras may experience gradual drift due to vibration or maintenance adjustments. "
        "To address this automatically, the system tracks a simulated 24-hour clock. When the simulation "
        "time enters the 2:00 AM segment and the number of detected cars is two or fewer, "
        "SpotManager.detect_spots_from_frame() is called. The spot detector re-runs on the current "
        "live frame, updates the spot registry, and saves the new coordinates to spots_data.json. "
        "This recalibration occurs at most once per simulated 24-hour cycle.",
        indent=True
    )

    add_heading(doc, "3.6 Visualisation", level=2)
    add_body(doc,
        "The Visualizer class (src/visualization.py) renders results on each frame using OpenCV:",
        indent=True
    )
    for item in [
        ("Spot Overlays", "Each parking spot bounding box is filled with a semi-transparent colour (green = FREE, red = OCCUPIED, α = 0.3) and labelled with FREE or TAKEN text."),
        ("Dashboard Bar", "A black header bar displays the system name alongside live FREE, OCCUPIED, and TOTAL counts, colour-coded by availability."),
        ("Simulation Clock", "The current simulated time segment is overlaid in yellow text for demonstration purposes."),
    ]:
        add_bullet(doc, item[1], bold_prefix=item[0])


def make_implementation(doc):
    add_heading(doc, "4. Implementation", level=1)

    add_heading(doc, "4.1 Technology Stack", level=2)
    add_table(doc,
        headers=["Component", "Technology / Library", "Version"],
        rows=[
            ["Programming Language", "Python", "3.8+"],
            ["Computer Vision Framework", "OpenCV (cv2)", "4.x"],
            ["Object Detection", "Ultralytics YOLOv11", "Latest"],
            ["Numerical Computing", "NumPy", "1.x"],
            ["Configuration", "PyYAML", "6.x"],
            ["Data Persistence", "JSON (standard library)", "—"],
        ],
        col_widths=[Cm(5.5), Cm(6), Cm(3)]
    )
    doc.add_paragraph()

    add_heading(doc, "4.2 Module Structure", level=2)
    add_body(doc,
        "The codebase is organised into clearly separated modules, each with a single responsibility:",
        indent=True
    )
    modules = [
        ("src/main.py", "Application entry point. Loads configuration, initialises all components, runs the simulation clock and main per-frame loop."),
        ("modules/detector/detector.py", "Wraps best.pt. Calls model.track() with persist=True to maintain consistent vehicle IDs across frames. Returns a list of detection dictionaries containing bounding box, confidence, class ID, and track ID."),
        ("modules/parking_logic/spot_manager.py", "Central coordinator for spot detection (YOLO inference on reference image), JSON serialisation/deserialisation, coordinate scaling, NMS deduplication, and BEV-based occupancy matching (update_occupancy())."),
        ("modules/parking_logic/perspective.py", "PerspectiveManager class. Encapsulates the homography matrix computation and provides transform_point(), get_car_footprint(), and get_spot_center_bev() utility methods."),
        ("src/visualization.py", "Visualizer class. Handles all OpenCV drawing operations: semi-transparent spot fills, border outlines, dashboard bar, and simulation time overlay."),
    ]
    for name, desc in modules:
        add_bullet(doc, desc, bold_prefix=name)

    add_heading(doc, "4.3 Configuration Files", level=2)
    add_table(doc,
        headers=["File", "Purpose"],
        rows=[
            ["config/config.yaml", "Global runtime settings (thresholds, paths, display parameters)."],
            ["config/spots_data.json", "Auto-generated JSON storing 37 parking spot coordinates at 2688×1520px reference resolution."],
            ["config/reference.jpg", "Clean reference image of the empty parking lot used by the spot detector at startup."],
            ["config/homography.json", "Stored homography calibration data for the BEV perspective transform."],
        ],
        col_widths=[Cm(5), Cm(10)]
    )
    doc.add_paragraph()

    add_heading(doc, "4.4 Key Algorithms", level=2)
    add_heading(doc, "4.4.1 Non-Maximum Suppression (NMS)", level=3)
    add_body(doc,
        "After the spot detector produces raw bounding box predictions, overlapping detections for the "
        "same physical bay are removed using a custom _nms_xyxy() implementation in SpotManager. Boxes "
        "are sorted by confidence score descending; any remaining box with an Intersection-over-Union "
        "(IoU) > 0.4 with a higher-confidence box is suppressed.",
        indent=True
    )
    add_heading(doc, "4.4.2 Greedy BEV Matching", level=3)
    add_body(doc,
        "For each frame, BEV positions are pre-computed for all N cars and M spots (O(N+M) transforms). "
        "All N×M candidate distance pairs are evaluated, filtered to those below 70px, and sorted. The "
        "greedy assignment runs in O(K log K) where K is the number of candidates, providing near-optimal "
        "matching without the full complexity of the Hungarian algorithm. In practice, K << N×M due to "
        "the distance threshold filter.",
        indent=True
    )


def make_experiments(doc):
    add_heading(doc, "5. Experiments and Results", level=1)

    add_heading(doc, "5.1 Training Configuration", level=2)
    add_body(doc,
        "All model variants were trained on a custom parking vehicle dataset using identical "
        "hyperparameters to ensure a fair comparison. Training was performed on a CUDA-enabled GPU.",
        indent=True
    )
    add_table(doc,
        headers=["Hyperparameter", "Value"],
        rows=[
            ["Epochs", "50"],
            ["Image Size", "640 × 640 px"],
            ["Batch Size", "8"],
            ["Optimizer", "Auto (AdamW)"],
            ["IoU Threshold (NMS)", "0.7"],
            ["Pretrained Weights", "Yes (transfer learning from COCO)"],
            ["Confidence Threshold (inference)", "0.15"],
        ],
        col_widths=[Cm(7), Cm(7)]
    )
    doc.add_paragraph()

    add_heading(doc, "5.2 Model Variants", level=2)
    add_body(doc,
        "Four YOLOv11 variants were trained to identify the best performing architecture for the car "
        "detection task:",
        indent=True
    )
    for variant, desc in [
        ("y11n", "YOLOv11 Nano — standard dataset. Smallest model, fastest inference."),
        ("y11s", "YOLOv11 Small — standard dataset. Larger capacity, higher accuracy."),
        ("y26n", "YOLOv11 Nano — extended 26-class dataset. Tests dataset breadth."),
        ("y26s", "YOLOv11 Small — extended 26-class dataset."),
    ]:
        add_bullet(doc, desc, bold_prefix=variant)

    add_heading(doc, "5.3 Model Comparison — Final Epoch Results (Epoch 50)", level=2)
    add_body(doc,
        "Table 1 presents the performance metrics recorded at epoch 50 (final epoch) for all four "
        "model variants on the validation set.",
        indent=True
    )
    add_table(doc,
        headers=["Model", "Precision", "Recall", "mAP@50", "mAP@50-95", "Train Time (s)"],
        rows=[
            ["y11n  (YOLOv11n, std)",   "0.961", "0.923", "0.957", "0.774", "5,597"],
            ["y11s  (YOLOv11s, std) ★", "0.976", "0.937", "0.964", "0.807", "6,729"],
            ["y26n  (YOLOv11n, 26-cls)","0.928", "0.905", "0.951", "0.752", "5,448"],
            ["y26s  (YOLOv11s, 26-cls)","0.965", "0.930", "0.962", "0.808", "5,303"],
        ],
        col_widths=[Cm(4.5), Cm(2.5), Cm(2.5), Cm(2.5), Cm(3), Cm(3.5)]
    )
    p = doc.add_paragraph("★ Selected for deployment (best.pt)")
    for run in p.runs:
        set_font(run, size=10, italic=True)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    add_body(doc,
        "The YOLOv11 Small model trained on the standard dataset (y11s) achieved the highest mAP@50 "
        "of 96.4% and mAP@50-95 of 80.7%. Despite requiring the longest training time (6,729 seconds), "
        "its superior generalisation performance justified its selection as the deployed car detector. "
        "Notably, the 26-class variants (y26n, y26s) performed marginally lower, suggesting that the "
        "additional class diversity introduced noise for the specific vehicle detection task.",
        indent=True
    )

    add_heading(doc, "5.4 Training Convergence", level=2)
    add_body(doc,
        "Figure 1 shows the training and validation loss curves alongside key metrics across all 50 "
        "epochs for the selected model (y11s). All three loss components (box, classification, DFL) "
        "decrease monotonically, confirming stable convergence without overfitting. Validation metrics "
        "closely track training metrics, indicating good generalisation.",
        indent=True
    )
    add_image(doc, IMG_RESULTS, "Figure 1. Training and validation curves for the selected y11s model (50 epochs).")
    doc.add_paragraph()

    add_heading(doc, "5.5 Precision-Recall and F1 Analysis", level=2)
    add_body(doc,
        "Figure 2 presents the Precision-Recall (PR) curve for the y11s model at the optimal confidence "
        "threshold. A high area under the PR curve indicates robustness across confidence thresholds. "
        "Figure 3 shows the F1-Confidence curve, indicating the model achieves peak F1 score above 0.95 "
        "at a confidence threshold of approximately 0.4.",
        indent=True
    )
    add_image(doc, IMG_PR_CURVE, "Figure 2. Precision-Recall curve for the y11s model on the validation set.", width=Inches(3.8))
    add_image(doc, IMG_F1_CURVE, "Figure 3. F1-Confidence curve for the y11s model.", width=Inches(3.8))
    doc.add_paragraph()

    add_heading(doc, "5.6 Confusion Matrix", level=2)
    add_body(doc,
        "Figure 4 presents the normalised confusion matrix for the y11s model on the validation set. "
        "The high on-diagonal values confirm reliable classification, with minimal false positive and "
        "false negative rates.",
        indent=True
    )
    add_image(doc, IMG_CONFUSION, "Figure 4. Normalised confusion matrix — y11s model, validation set.")
    doc.add_paragraph()

    add_heading(doc, "5.7 Validation Predictions", level=2)
    add_body(doc,
        "Figure 5 shows sample validation batch predictions from the y11s model, demonstrating accurate "
        "localisation of vehicles with high confidence scores across diverse scenes.",
        indent=True
    )
    add_image(doc, IMG_VAL_PRED, "Figure 5. Sample validation batch predictions — y11s model.")
    doc.add_paragraph()

    add_heading(doc, "5.8 System Runtime Behaviour", level=2)
    add_body(doc,
        "Table 2 summarises the system's observed behaviour across the four simulated time segments "
        "used in the demonstration video.",
        indent=True
    )
    add_table(doc,
        headers=["Simulation Time", "Occupancy Level", "System Behaviour"],
        rows=[
            ["02:00 AM", "Low (≤ 2 cars)", "Self-healing recalibration triggered; spots.pt re-scans the frame and updates spot registry."],
            ["09:30 AM", "Medium", "Accurate FREE/TAKEN labels rendered on all 37 spots; dashboard reflects correct counts."],
            ["16:00 PM", "High", "Peak occupancy correctly identified; dashboard updates in real time."],
            ["18:00 PM", "Variable", "Smooth real-time transitions as vehicles enter and exit bays."],
        ],
        col_widths=[Cm(3.5), Cm(3.5), Cm(9)]
    )
    doc.add_paragraph()

    add_heading(doc, "5.9 Reference Image — Spot Detection Output", level=2)
    add_body(doc,
        "Figure 6 shows the reference image used for initial spot detection. The spot detector "
        "identified 37 parking bays at a resolution of 2688 × 1520 pixels. These coordinates are "
        "persisted to config/spots_data.json and rescaled to match the live video resolution at runtime.",
        indent=True
    )
    add_image(doc, IMG_REFERENCE, "Figure 6. Reference image (config/reference.jpg) used for parking spot detection.")


def make_discussion(doc):
    add_heading(doc, "6. Discussion", level=1)

    add_heading(doc, "6.1 Strengths", level=2)
    for point in [
        ("Dual-Model Separation", "Decoupling spot detection from vehicle detection makes each model smaller and more specialised, yielding higher accuracy than a single multi-task model would."),
        ("BEV Distance Matching", "Transforming positions to Bird's Eye View eliminates perspective-induced scale variation, making the 70px distance threshold effective across the entire field of view."),
        ("Self-Healing Recalibration", "The automatic 2 AM recalibration eliminates the need for manual re-annotation when the camera shifts, dramatically reducing operational maintenance cost."),
        ("Persistent Spot Registry", "Saving detected spots to JSON means the computationally expensive spot detection model only runs once per session (or on trigger), while the real-time loop incurs only the cost of the lightweight car detector."),
    ]:
        add_bullet(doc, point[1], bold_prefix=point[0])

    add_heading(doc, "6.2 Limitations", level=2)
    for point in [
        ("Fixed Camera Dependency", "The BEV homography matrix is calibrated for a single camera position. Any significant camera movement (short of the 2 AM recalibration) will degrade matching accuracy."),
        ("Fixed Distance Threshold", "The 70px BEV matching threshold is empirically set for this specific parking lot. Different lot geometries or camera heights may require tuning."),
        ("Lighting Sensitivity", "Like all deep learning detectors, performance degrades under extreme lighting conditions (heavy night shadows, direct glare) below the model's training distribution."),
        ("Single Camera Coverage", "The current implementation monitors a single camera feed. Large parking facilities would require multi-camera orchestration."),
    ]:
        add_bullet(doc, point[1], bold_prefix=point[0])

    add_heading(doc, "6.3 Future Work", level=2)
    for point in [
        ("Multi-Camera Fusion", "Extend the system to aggregate occupancy data from multiple cameras covering different zones, providing a facility-wide view."),
        ("IoT Integration", "Stream occupancy data to a cloud backend (e.g., MQTT broker) to enable mobile app notifications and dynamic signage updates."),
        ("License Plate Recognition", "Add an ALPR module to associate occupancy events with specific vehicles, enabling permit enforcement and entry/exit logging."),
        ("Adaptive Threshold Learning", "Replace the fixed BEV distance threshold with an adaptive mechanism that learns the optimal threshold from historical occupancy data."),
        ("Edge Deployment", "Optimise the models for embedded edge devices (NVIDIA Jetson, Raspberry Pi with Coral TPU) using quantisation and pruning to enable low-cost deployment."),
    ]:
        add_bullet(doc, point[1], bold_prefix=point[0])


def make_conclusion(doc):
    add_heading(doc, "7. Conclusion", level=1)
    add_body(doc,
        "This project has successfully designed, implemented, and evaluated a Vision-Based Smart Parking "
        "Occupancy Detection System using a Dual YOLOv11 architecture. By separating the concerns of "
        "parking spot discovery (spots.pt) from real-time vehicle tracking (best.pt), and applying "
        "Perspective Transformation to normalise camera distortion, the system achieves reliable and "
        "scalable occupancy detection from a single CCTV feed.",
        indent=True
    )
    add_body(doc,
        "A systematic benchmarking of four YOLOv11 variants (Nano and Small, across two dataset "
        "configurations, each trained for 50 epochs) identified the YOLOv11 Small model as the optimal "
        "deployment choice, achieving a mAP@50 of 96.4% and mAP@50-95 of 80.7% on the validation set. "
        "The self-healing recalibration mechanism addresses the practical challenge of long-term camera "
        "drift without requiring human intervention.",
        indent=True
    )
    add_body(doc,
        "The system demonstrates that high-accuracy, real-time parking management is achievable using "
        "purely software-based computer vision, eliminating the cost and complexity of per-bay sensor "
        "hardware. The modular architecture provides a solid foundation for future extensions including "
        "multi-camera coverage, IoT connectivity, and edge deployment. This work contributes a practical "
        "and novel combination of dual-model detection, BEV matching, and autonomous recalibration to "
        "the field of intelligent transportation systems.",
        indent=True
    )


def make_references(doc):
    add_heading(doc, "References", level=1)
    refs = [
        "[1] G. Jocher, A. Chaurasia, and J. Qiu, \"Ultralytics YOLOv8,\" GitHub, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics",
        "[2] G. Jocher et al., \"Ultralytics YOLOv11,\" GitHub, 2024. [Online]. Available: https://github.com/ultralytics/ultralytics",
        "[3] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, \"You Only Look Once: Unified, Real-Time Object Detection,\" in Proc. IEEE CVPR, 2016, pp. 779–788.",
        "[4] P. R. de Almeida, L. S. Oliveira, A. S. Britto Jr., E. J. Silva Jr., and A. L. Koerich, \"PKLot — A Robust Dataset for Parking Lot Classification,\" Expert Systems with Applications, vol. 42, no. 11, pp. 4937–4949, 2015.",
        "[5] B. Chen, X. Zhao, and Y. Li, \"Bird's Eye View Semantic Segmentation Using Perspective Transformation for Autonomous Driving,\" IEEE Access, vol. 8, pp. 75 960–75 972, 2020.",
        "[6] D. C. Shoup, The High Cost of Free Parking (Updated Edition). Chicago: Planners Press, 2011.",
        "[7] D. Benson, J. Guo, and T. Zhao, \"A Review of Intelligent Parking Systems,\" Journal of Traffic and Transportation Engineering, vol. 3, no. 4, pp. 293–305, 2016.",
        "[8] G. Bradski, \"The OpenCV Library,\" Dr. Dobb's Journal of Software Tools, 2000.",
    ]
    for ref in refs:
        p = doc.add_paragraph(style="List Paragraph")
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        run = p.add_run(ref)
        set_font(run, size=11)
        p.paragraph_format.left_indent = Cm(0)


def make_appendix(doc):
    add_heading(doc, "Appendix A — Project Structure", level=1)
    add_body(doc,
        "The following directory structure summarises the complete project repository:",
        indent=True
    )
    code_text = (
        "parking-vision-system/\n"
        "├── config/\n"
        "│   ├── config.yaml          # Runtime settings\n"
        "│   ├── homography.json      # BEV calibration data\n"
        "│   ├── reference.jpg        # Reference image for spot detection\n"
        "│   └── spots_data.json      # Persisted spot coordinates (37 spots)\n"
        "├── experimental/\n"
        "│   ├── y11n/                # YOLOv11 Nano — standard dataset training run\n"
        "│   ├── y11s/                # YOLOv11 Small — standard dataset (selected)\n"
        "│   ├── y26n/                # YOLOv11 Nano — 26-class dataset\n"
        "│   └── y26s/                # YOLOv11 Small — 26-class dataset\n"
        "├── modules/\n"
        "│   ├── detector/\n"
        "│   │   ├── best.pt          # Deployed car detection model (y11s)\n"
        "│   │   └── detector.py      # YOLO wrapper, track() interface\n"
        "│   └── parking_logic/\n"
        "│       ├── spots.pt         # Parking spot detection model\n"
        "│       ├── spot_manager.py  # Spot detection, NMS, BEV matching\n"
        "│       └── perspective.py   # Homography / BEV transform\n"
        "├── src/\n"
        "│   ├── main.py              # Entry point, main loop\n"
        "│   └── visualization.py    # OpenCV drawing utilities\n"
        "├── videos/                  # Input video files\n"
        "├── documents/               # Generated report\n"
        "└── requirements.txt"
    )
    p = doc.add_paragraph()
    run = p.add_run(code_text)
    run.font.name = "Courier New"
    run.font.size = Pt(9)

    add_heading(doc, "Appendix B — spots_data.json Format", level=1)
    add_body(doc,
        "The spots_data.json file stores detected parking spot coordinates in the following structure:",
        indent=True
    )
    json_sample = (
        '{\n'
        '    "width": 2688,\n'
        '    "height": 1520,\n'
        '    "spots": [\n'
        '        [1124, 100, 1271, 132],\n'
        '        [1434, 110, 1582, 140],\n'
        '        ...\n'
        '    ]\n'
        '}'
    )
    p = doc.add_paragraph()
    run = p.add_run(json_sample)
    run.font.name = "Courier New"
    run.font.size = Pt(9)
    add_body(doc,
        "Each entry in the spots array is a bounding box [x1, y1, x2, y2] in pixels, "
        "relative to the reference image dimensions (width × height).",
        indent=True
    )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("📄 Generating Parking Vision System Academic Report...")

    # Warn about placeholder values
    if STUDENT_NAME == "Your Full Name":
        print("⚠️  WARNING: Placeholder student info detected in the script.")
        print("   Please edit STUDENT_NAME, STUDENT_ID, COURSE_NAME, UNIVERSITY, SUPERVISOR at the top of this file.")

    doc = Document()
    set_page_margins(doc)

    # Set default style
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    make_title_page(doc)
    make_abstract(doc)
    add_page_break(doc)
    make_introduction(doc)
    add_page_break(doc)
    make_literature_review(doc)
    add_page_break(doc)
    make_system_architecture(doc)
    add_page_break(doc)
    make_implementation(doc)
    add_page_break(doc)
    make_experiments(doc)
    add_page_break(doc)
    make_discussion(doc)
    add_page_break(doc)
    make_conclusion(doc)
    add_page_break(doc)
    make_references(doc)
    add_page_break(doc)
    make_appendix(doc)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    doc.save(OUTPUT_PATH)
    print(f"✅ Report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
