"""
generate_slides.py  —  Parking Vision System Presentation
Minimal style: white background, Crimson Red (#DC143C) primary colour.
Run: python generate_slides.py
Output: documents/Parking_Vision_System_Presentation.pptx
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── Palette ──────────────────────────────────────────────────────────────────
CRIMSON   = RGBColor(0xDC, 0x14, 0x3C)   # #DC143C
DARK      = RGBColor(0x1A, 0x1A, 0x2E)   # near-black for body text
GREY      = RGBColor(0x60, 0x60, 0x70)   # muted grey for sub-text
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
LIGHTGREY = RGBColor(0xF5, 0xF5, 0xF7)   # slide accent bg panels

# ── Slide dimensions (widescreen 16:9) ───────────────────────────────────────
W = Inches(13.33)
H = Inches(7.5)

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H

BLANK_LAYOUT = prs.slide_layouts[6]   # completely blank

# ═══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════════

def add_rect(slide, l, t, w, h, fill_rgb=None, line_rgb=None, line_width_pt=0):
    """Add a filled rectangle shape."""
    shape = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    shape.line.fill.background()
    if fill_rgb:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_rgb
    else:
        shape.fill.background()
    if line_rgb and line_width_pt:
        shape.line.color.rgb = line_rgb
        shape.line.width = Pt(line_width_pt)
    else:
        shape.line.fill.background()
    return shape


def add_textbox(slide, text, l, t, w, h,
                font_size=18, bold=False, color=DARK,
                align=PP_ALIGN.LEFT, italic=False, wrap=True):
    txb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    txb.word_wrap = wrap
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size   = Pt(font_size)
    run.font.bold   = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name   = "Calibri"
    return txb


def add_crimson_bar(slide, height_in=0.06):
    """Thin crimson accent line across the top."""
    add_rect(slide, 0, 0, 13.33, height_in, fill_rgb=CRIMSON)


def add_slide_number(slide, num):
    add_textbox(slide, str(num), 12.5, 7.1, 0.7, 0.3,
                font_size=9, color=GREY, align=PP_ALIGN.RIGHT)


def add_footer_line(slide):
    """Thin grey separator above slide number."""
    add_rect(slide, 0.4, 7.0, 12.53, 0.02, fill_rgb=GREY)


def slide_heading(slide, title, subtitle=None):
    """Standard section heading with crimson underline."""
    add_textbox(slide, title, 0.5, 0.2, 12.0, 0.65,
                font_size=28, bold=True, color=CRIMSON, align=PP_ALIGN.LEFT)
    add_rect(slide, 0.5, 0.9, 1.6, 0.045, fill_rgb=CRIMSON)   # underline accent
    if subtitle:
        add_textbox(slide, subtitle, 0.5, 0.95, 12.0, 0.35,
                    font_size=13, color=GREY, align=PP_ALIGN.LEFT)


def bullet_lines(slide, items, l, t, w, h,
                 font_size=14, color=DARK, spacing=0.38, marker="•"):
    for i, item in enumerate(items):
        add_textbox(slide, f"{marker}  {item}",
                    l, t + i * spacing, w, spacing + 0.05,
                    font_size=font_size, color=color)


# ═══════════════════════════════════════════════════════════════════════════════
# Slide 1 — Title slide
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK_LAYOUT)
# Full-width crimson top bar (taller on title)
add_rect(slide, 0, 0, 13.33, 0.08, fill_rgb=CRIMSON)
# White background already default

# Main title
add_textbox(slide,
            "Vision-Based Smart Parking\nOccupancy Detection System",
            1.0, 1.5, 11.5, 2.0,
            font_size=40, bold=True, color=DARK, align=PP_ALIGN.CENTER)

add_rect(slide, 4.6, 3.55, 4.1, 0.06, fill_rgb=CRIMSON)   # separator

add_textbox(slide, "Using Dual YOLOv11 Architecture",
            1.0, 3.65, 11.5, 0.55,
            font_size=20, bold=False, color=CRIMSON, align=PP_ALIGN.CENTER, italic=True)

# Meta block
meta = [
    ("Group:", "Group [X]"),
    ("Members:", "[Your Names]"),
    ("Advisor:", "[Advisor Name]"),
    ("Course:", "[Course Name]  |  March 2026"),
]
for idx, (label, value) in enumerate(meta):
    y = 4.5 + idx * 0.47
    add_textbox(slide, label, 3.5, y, 1.5, 0.45,
                font_size=13, bold=True, color=CRIMSON)
    add_textbox(slide, value, 4.95, y, 6.0, 0.45,
                font_size=13, color=DARK)

# Bottom bar
add_rect(slide, 0, 7.1, 13.33, 0.4, fill_rgb=CRIMSON)
add_textbox(slide, "UNIVERSITY PROJECT  |  2026",
            0, 7.12, 13.33, 0.35,
            font_size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════════════════════
# Slide 2 — Background / Problem
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK_LAYOUT)
add_crimson_bar(slide)
slide_heading(slide, "Background — The Parking Problem")

points = [
    "Drivers waste an average of 17 minutes per trip searching for parking",
    "Excessive cruising contributes to urban traffic congestion & carbon emissions",
    "Parking operators lack real-time occupancy data, reducing facility efficiency",
    "Affordable, scalable monitoring remains a critical unmet need in smart cities",
]
bullet_lines(slide, points, 0.5, 1.25, 8.5, 4.0, font_size=15)

# Stats panel (right)
add_rect(slide, 9.5, 1.2, 3.4, 5.6, fill_rgb=LIGHTGREY)
stats = [("17 min", "avg search time\nper trip"),
         ("30%", "of urban traffic\nis cruising for parking"),
         ("$1,000+", "annual cost per\nsensor per bay")]
for i, (num, lbl) in enumerate(stats):
    y = 1.5 + i * 1.85
    add_textbox(slide, num, 9.6, y, 3.2, 0.7,
                font_size=30, bold=True, color=CRIMSON, align=PP_ALIGN.CENTER)
    add_textbox(slide, lbl, 9.6, y + 0.65, 3.2, 0.65,
                font_size=11, color=GREY, align=PP_ALIGN.CENTER)

add_footer_line(slide)
add_slide_number(slide, 2)

# ═══════════════════════════════════════════════════════════════════════════════
# Slide 3 — Current Methods & Limitations
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK_LAYOUT)
add_crimson_bar(slide)
slide_heading(slide, "Current Methods & Limitations")

methods = [
    ("Manual Monitoring",
     "Security personnel conduct physical walkthroughs — not scalable, error-prone"),
    ("Per-Bay Sensors",
     "Magnetic / ultrasonic sensors per space — high cost, difficult to maintain"),
    ("Single-Model Vision",
     "One CNN for all tasks — lower accuracy, struggles with multi-class inference"),
    ("Perspective Distortion",
     "Far objects appear smaller; naïve pixel-distance matching fails"),
    ("Manual Spot Annotation",
     "Parking bay coordinates must be hand-labelled — labour intensive, brittle"),
]
for i, (title, desc) in enumerate(methods):
    y = 1.2 + i * 1.06
    add_rect(slide, 0.5, y, 12.3, 0.9, fill_rgb=LIGHTGREY)
    add_textbox(slide, title,  0.65, y + 0.05, 3.0, 0.42,
                font_size=13, bold=True, color=CRIMSON)
    add_textbox(slide, desc,   3.65, y + 0.05, 9.0, 0.42,
                font_size=12, color=DARK)
    # small crimson left accent edge
    add_rect(slide, 0.5, y, 0.06, 0.9, fill_rgb=CRIMSON)

add_footer_line(slide)
add_slide_number(slide, 3)

# ═══════════════════════════════════════════════════════════════════════════════
# Slide 4 — Key Insights ("Aha" slide)
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK_LAYOUT)
add_crimson_bar(slide)
slide_heading(slide, "Key Insights")
add_textbox(slide, "Four ideas that define our approach",
            0.5, 0.9, 10.0, 0.35, font_size=13, color=GREY, italic=True)

insights = [
    ("01", "One Camera, Many Slots",
     "A single overhead camera can monitor an entire parking zone,\neliminating per-bay sensor hardware."),
    ("02", "Separate the Detection Tasks",
     "Spot detection and car detection are different problems—\nusing dedicated models for each yields higher accuracy."),
    ("03", "BEV Fixes Perspective",
     "Bird's Eye View projection normalises distance distortion,\nmaking proximity matching reliable across the entire lot."),
    ("04", "Recalibration Reduces Maintenance",
     "Automated 2 AM spot re-scan self-heals after camera drift,\neliminating the need for human re-annotation."),
]
cols = [(0.4, 1.35), (6.8, 1.35), (0.4, 4.15), (6.8, 4.15)]
for (cx, cy), (num, title, desc) in zip(cols, insights):
    add_rect(slide, cx, cy, 6.1, 2.55, fill_rgb=LIGHTGREY)
    add_textbox(slide, num,   cx+0.15, cy+0.12, 1.0, 0.6,
                font_size=28, bold=True, color=CRIMSON)
    add_textbox(slide, title, cx+0.15, cy+0.68, 5.7, 0.42,
                font_size=14, bold=True, color=DARK)
    add_textbox(slide, desc,  cx+0.15, cy+1.1,  5.7, 1.2,
                font_size=11, color=GREY)

add_footer_line(slide)
add_slide_number(slide, 4)

# ═══════════════════════════════════════════════════════════════════════════════
# Slide 5 — System Overview (pipeline diagram)
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK_LAYOUT)
add_crimson_bar(slide)
slide_heading(slide, "System Overview")

# --- pipeline boxes ---
pipeline = [
    ("Reference\nImage", 0.4, 2.2),
    ("Spot\nDetector\n(spots.pt)", 2.3, 2.2),
    ("Spot\nCoordinates\nJSON", 4.3, 2.2),
    ("BEV\nTransform", 6.2, 2.2),
    ("Matching\nAlgorithm", 8.1, 2.2),
    ("Occupancy\nOutput", 10.2, 2.2),
]
inputs_row = [
    ("Live\nVideo", 6.2, 4.0),
    ("Car Detector\n(best.pt)\n+ Tracker", 8.1, 4.0),
]

BOX_W, BOX_H = 1.7, 1.3

for label, bx, by in pipeline:
    is_model = "pt)" in label
    fill = CRIMSON if is_model else LIGHTGREY
    txt_col = WHITE if is_model else DARK
    add_rect(slide, bx, by, BOX_W, BOX_H, fill_rgb=fill)
    add_textbox(slide, label, bx, by + 0.2, BOX_W, BOX_H - 0.1,
                font_size=11, bold=is_model, color=txt_col, align=PP_ALIGN.CENTER)
    # Arrow (simple right-pointing box) — skip last
    if label != "Occupancy\nOutput":
        add_textbox(slide, "→", bx + BOX_W, by + 0.4, 0.35, 0.5,
                    font_size=18, color=GREY, align=PP_ALIGN.CENTER)

for label, bx, by in inputs_row:
    is_model = "pt)" in label
    fill = CRIMSON if is_model else LIGHTGREY
    txt_col = WHITE if is_model else DARK
    add_rect(slide, bx, by, BOX_W, BOX_H, fill_rgb=fill)
    add_textbox(slide, label, bx, by + 0.1, BOX_W, BOX_H,
                font_size=11, bold=is_model, color=txt_col, align=PP_ALIGN.CENTER)
    if label != "Car Detector\n(best.pt)\n+ Tracker":
        add_textbox(slide, "→", bx + BOX_W, by + 0.4, 0.35, 0.5,
                    font_size=18, color=GREY, align=PP_ALIGN.CENTER)

# vertical arrow from car detector to BEV
add_textbox(slide, "↑", 6.95, 3.6, 0.5, 0.6, font_size=18, color=GREY, align=PP_ALIGN.CENTER)

add_textbox(slide, "★  Red boxes = YOLOv11 model inference",
            0.4, 5.65, 7.0, 0.4, font_size=11, color=CRIMSON, italic=True)

add_footer_line(slide)
add_slide_number(slide, 5)

# ═══════════════════════════════════════════════════════════════════════════════
# Slide 6 — Core Method (technical details)
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK_LAYOUT)
add_crimson_bar(slide)
slide_heading(slide, "Core Technical Method")

sections = [
    ("Dual YOLOv11 Models",
     ["spots.pt — detects empty bay boundaries from a reference image",
      "best.pt — tracks vehicles frame-by-frame in live video",
      "Each model is specialised, smaller, and more accurate than a single multi-task model"]),
    ("Homography / Bird's Eye View",
     ["4 anchor points map camera view → flat top-down canvas (400 × 800 px)",
      "Car's tyre contact point & spot centre both projected to BEV",
      "Eliminates perspective-induced scale variation between near/far spots"]),
    ("Distance-Based Greedy Matching",
     ["All (car, spot) BEV distance pairs computed each frame",
      "Sorted ascending — closest pairs assigned first (greedy, O(K log K))",
      "70 px BEV threshold rejects implausible matches"]),
    ("Self-Healing Recalibration",
     ["Simulated clock triggers spot re-scan at 2:00 AM",
      "Only fires when ≤ 2 cars detected (low traffic window)",
      "Re-saves spots_data.json — no human annotation required"]),
]

for i, (hdr, bullets) in enumerate(sections):
    col = 0 if i < 2 else 1
    row = i % 2
    lx = 0.4 + col * 6.55
    ly = 1.25 + row * 2.9
    add_rect(slide, lx, ly, 6.1, 2.6, fill_rgb=LIGHTGREY)
    add_rect(slide, lx, ly, 0.06, 2.6, fill_rgb=CRIMSON)
    add_textbox(slide, hdr, lx + 0.2, ly + 0.1, 5.7, 0.42,
                font_size=14, bold=True, color=CRIMSON)
    for j, b in enumerate(bullets):
        add_textbox(slide, f"• {b}", lx + 0.2, ly + 0.58 + j * 0.62, 5.7, 0.65,
                    font_size=11, color=DARK)

add_footer_line(slide)
add_slide_number(slide, 6)

# ═══════════════════════════════════════════════════════════════════════════════
# Slide 7 — Implementation / Tools
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK_LAYOUT)
add_crimson_bar(slide)
slide_heading(slide, "Implementation & Tools")

tech = [
    ("Python 3.8+",        "Core programming language"),
    ("OpenCV 4.x",         "Video capture, BEV transform, visualisation"),
    ("Ultralytics YOLOv11","Object detection & real-time tracking"),
    ("NumPy",              "Numerical arrays & distance computation"),
    ("PyYAML",             "Configuration file loading (config.yaml)"),
    ("JSON stdlib",        "Persistent spot coordinate storage"),
]
for i, (tool, desc) in enumerate(tech):
    y = 1.25 + i * 0.72
    add_rect(slide, 0.4, y, 5.6, 0.62, fill_rgb=LIGHTGREY)
    add_rect(slide, 0.4, y, 0.06, 0.62, fill_rgb=CRIMSON)
    add_textbox(slide, tool, 0.6, y + 0.08, 2.1, 0.48, font_size=13, bold=True, color=CRIMSON)
    add_textbox(slide, desc, 2.75, y + 0.1, 3.1, 0.45, font_size=12, color=DARK)

# Module structure box
add_rect(slide, 6.5, 1.2, 6.4, 5.6, fill_rgb=LIGHTGREY)
add_textbox(slide, "Module Structure", 6.65, 1.25, 5.9, 0.45,
            font_size=14, bold=True, color=CRIMSON)
modules = [
    "src/main.py           — Entry point & main loop",
    "modules/detector/     — YOLOv11 car detector wrapper",
    "modules/parking_logic/",
    "  ├ spot_manager.py   — Spot detection, NMS, BEV matching",
    "  └ perspective.py    — Homography / BEV transform",
    "src/visualization.py  — OpenCV drawing utilities",
    "config/config.yaml    — Runtime settings",
    "config/spots_data.json— 37 persisted spot coords",
    "config/reference.jpg  — Empty lot reference image",
]
for i, line in enumerate(modules):
    add_textbox(slide, line, 6.65, 1.75 + i * 0.53, 6.0, 0.52,
                font_size=10, color=DARK, italic=line.startswith("  "))

add_footer_line(slide)
add_slide_number(slide, 7)

# ═══════════════════════════════════════════════════════════════════════════════
# Slide 8 — Model Training
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK_LAYOUT)
add_crimson_bar(slide)
slide_heading(slide, "Model Training")

# Hyperparameter table (left)
add_textbox(slide, "Training Configuration", 0.4, 1.2, 5.5, 0.4,
            font_size=14, bold=True, color=DARK)
hyperparams = [
    ("Epochs",             "50"),
    ("Image Size",         "640 × 640 px"),
    ("Batch Size",         "8"),
    ("Optimizer",          "AdamW (auto)"),
    ("IoU Threshold",      "0.7"),
    ("Pre-trained",        "Yes — COCO weights"),
    ("Conf Threshold",     "0.15 (inference)"),
]
HDR_BG = CRIMSON
add_rect(slide, 0.4, 1.65, 3.2, 0.45, fill_rgb=CRIMSON)
add_textbox(slide, "Hyperparameter", 0.45, 1.67, 1.8, 0.4,
            font_size=11, bold=True, color=WHITE)
add_textbox(slide, "Value",          2.25, 1.67, 1.3, 0.4,
            font_size=11, bold=True, color=WHITE)
for i, (k, v) in enumerate(hyperparams):
    row_bg = LIGHTGREY if i % 2 == 0 else WHITE
    add_rect(slide, 0.4, 2.13 + i*0.5, 3.2, 0.48, fill_rgb=row_bg)
    add_textbox(slide, k, 0.5, 2.16 + i*0.5, 1.8, 0.42, font_size=11, color=DARK)
    add_textbox(slide, v, 2.25, 2.16 + i*0.5, 1.3, 0.42, font_size=11, color=DARK)

# Model variants (right)
add_textbox(slide, "Four Variants Trained", 4.1, 1.2, 8.8, 0.4,
            font_size=14, bold=True, color=DARK)
variants = [
    ("y11n", "YOLOv11 Nano",  "Standard dataset",  "Smallest / fastest"),
    ("y11s", "YOLOv11 Small", "Standard dataset",  "★ SELECTED — best accuracy"),
    ("y26n", "YOLOv11 Nano",  "26-class dataset",  "Tests class breadth"),
    ("y26s", "YOLOv11 Small", "26-class dataset",  "Comparison baseline"),
]
add_rect(slide, 4.1, 1.65, 8.8, 0.45, fill_rgb=CRIMSON)
for col_lbl, col_x, col_w in [("Model ID", 4.15, 1.0), ("Architecture", 5.2, 2.0),
                                ("Dataset", 7.25, 2.3), ("Notes", 9.6, 3.2)]:
    add_textbox(slide, col_lbl, col_x, 1.67, col_w, 0.4,
                font_size=11, bold=True, color=WHITE)

for i, (mid, arch, ds, note) in enumerate(variants):
    is_selected = "SELECTED" in note
    row_bg = RGBColor(0xFF, 0xEB, 0xEE) if is_selected else (LIGHTGREY if i%2==0 else WHITE)
    add_rect(slide, 4.1, 2.13+i*0.5, 8.8, 0.48, fill_rgb=row_bg)
    col_color = CRIMSON if is_selected else DARK
    for val, cx, cw in [(mid, 4.15, 1.0), (arch, 5.2, 2.0), (ds, 7.25, 2.3), (note, 9.6, 3.2)]:
        add_textbox(slide, val, cx, 2.16+i*0.5, cw, 0.42,
                    font_size=11, bold=is_selected, color=col_color)

# Same settings note
add_textbox(slide, "All models trained with identical hyperparameters for fair comparison.",
            0.4, 6.3, 12.5, 0.45, font_size=12, color=GREY, italic=True)

add_footer_line(slide)
add_slide_number(slide, 8)

# ═══════════════════════════════════════════════════════════════════════════════
# Slide 9 — Performance Results
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK_LAYOUT)
add_crimson_bar(slide)
slide_heading(slide, "Performance Results")

# Full comparison table
headers = ["Model", "Precision", "Recall", "mAP@50", "mAP@50-95", "Train Time (s)"]
rows = [
    ["y11n  (YOLOv11n, std)",   "0.961", "0.923", "95.7%", "77.4%", "5,597"],
    ["y11s  (YOLOv11s, std) ★", "0.976", "0.937", "96.4%", "80.7%", "6,729"],
    ["y26n  (YOLOv11n, 26-cls)","0.928", "0.905", "95.1%", "75.2%", "5,448"],
    ["y26s  (YOLOv11s, 26-cls)","0.965", "0.930", "96.2%", "80.8%", "5,303"],
]
col_xs = [0.4, 3.0, 4.7, 6.1, 7.6, 9.8]
col_ws = [2.55, 1.65, 1.35, 1.45, 2.15, 1.5]

add_rect(slide, 0.4, 1.25, 12.5, 0.48, fill_rgb=CRIMSON)
for hdr, cx, cw in zip(headers, col_xs, col_ws):
    add_textbox(slide, hdr, cx+0.05, 1.27, cw, 0.42,
                font_size=12, bold=True, color=WHITE)

for i, row in enumerate(rows):
    is_sel = "★" in row[0]
    bg = RGBColor(0xFF, 0xEB, 0xEE) if is_sel else (LIGHTGREY if i%2==0 else WHITE)
    add_rect(slide, 0.4, 1.76+i*0.55, 12.5, 0.53, fill_rgb=bg)
    for val, cx, cw in zip(row, col_xs, col_ws):
        add_textbox(slide, val, cx+0.05, 1.79+i*0.55, cw, 0.48,
                    font_size=12, bold=is_sel, color=CRIMSON if is_sel else DARK)

# Key metrics callout strip
kpis = [("96.4%", "mAP@50"), ("80.7%", "mAP@50-95"), ("37", "Spots\nDetected"),
        ("y11s", "Deployed\nModel")]
for i, (val, lbl) in enumerate(kpis):
    bx = 0.4 + i * 3.2
    add_rect(slide, bx, 4.1, 2.9, 1.8, fill_rgb=LIGHTGREY)
    add_rect(slide, bx, 4.1, 2.9, 0.06, fill_rgb=CRIMSON)
    add_textbox(slide, val, bx, 4.2, 2.9, 0.85,
                font_size=34, bold=True, color=CRIMSON, align=PP_ALIGN.CENTER)
    add_textbox(slide, lbl, bx, 5.0, 2.9, 0.7,
                font_size=12, color=GREY, align=PP_ALIGN.CENTER)

add_textbox(slide, "★ Selected for deployment",
            0.4, 6.05, 6.0, 0.35, font_size=11, color=CRIMSON, italic=True)

add_footer_line(slide)
add_slide_number(slide, 9)

# ═══════════════════════════════════════════════════════════════════════════════
# Slide 10 — System Behaviour / Demo
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK_LAYOUT)
add_crimson_bar(slide)
slide_heading(slide, "System Behaviour — Demo Timeline")

timeline = [
    ("02:00 AM", "Low Occupancy\n(≤ 2 cars)",
     "Self-healing recalibration triggered.\nspots.pt re-scans live frame,\nupdates spot registry & JSON."),
    ("09:30 AM", "Medium Occupancy",
     "Accurate FREE / TAKEN labels on all\n37 spots. Dashboard reflects\ncorrect counts in real time."),
    ("16:00 PM", "Peak Occupancy",
     "HIGH occupancy correctly identified.\nDashboard updates live as cars\nenter and occupy bays."),
    ("18:00 PM", "Variable Occupancy",
     "Smooth real-time transitions as\nvehicles enter and exit bays.\nGreedy matching resolves instantly."),
]
for i, (time, state, desc) in enumerate(timeline):
    bx = 0.4 + i * 3.22
    add_rect(slide, bx, 1.2, 3.0, 5.3, fill_rgb=LIGHTGREY)
    add_rect(slide, bx, 1.2, 3.0, 0.06, fill_rgb=CRIMSON)
    add_textbox(slide, time,  bx+0.1, 1.3,  2.8, 0.5,
                font_size=22, bold=True, color=CRIMSON, align=PP_ALIGN.CENTER)
    add_rect(slide, bx+0.3, 1.82, 2.4, 0.035, fill_rgb=GREY)
    add_textbox(slide, state, bx+0.1, 1.88, 2.8, 0.55,
                font_size=13, bold=True, color=DARK, align=PP_ALIGN.CENTER)
    add_textbox(slide, desc,  bx+0.15, 2.55, 2.7, 3.5,
                font_size=11, color=GREY, align=PP_ALIGN.CENTER)

add_footer_line(slide)
add_slide_number(slide, 10)

# ═══════════════════════════════════════════════════════════════════════════════
# Slide 11 — Current Progress / Limitations
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK_LAYOUT)
add_crimson_bar(slide)
slide_heading(slide, "Current Progress & Limitations")

done_items = [
    "YOLOv11 model training (4 variants benchmarked)",
    "Parking spot detection — spots.pt, NMS, JSON persistence",
    "Bird's Eye View homography transform",
    "Greedy BEV distance-based occupancy matching",
    "Self-healing 2 AM recalibration mechanism",
    "Real-time dashboard overlay (OpenCV)",
]
inprog_items = [
    "Demo video polishing & screen capture",
    "UI refinement — label styling, colour overlays",
]
future_items = [
    "Multi-camera orchestration",
    "IoT / MQTT cloud integration",
    "License plate recognition (ALPR)",
    "Edge deployment (Jetson / Coral TPU)",
]

col_data = [
    ("✅  Completed",   CRIMSON,                                  done_items,   0.4,  1.2, 4.3),
    ("🔄  In Progress", RGBColor(0xFF, 0x8C, 0x00),               inprog_items, 4.85, 1.2, 3.3),
    ("🔭  Future Work", RGBColor(0x00, 0x71, 0xBC),               future_items, 8.3,  1.2, 4.6),
]
for hdr, hdr_color, items, cx, cy, cw in col_data:
    add_rect(slide, cx, cy, cw, 5.4, fill_rgb=LIGHTGREY)
    add_rect(slide, cx, cy, cw, 0.06, fill_rgb=hdr_color)
    add_textbox(slide, hdr, cx+0.1, cy+0.1, cw-0.2, 0.48,
                font_size=14, bold=True, color=hdr_color)
    for j, item in enumerate(items):
        add_textbox(slide, f"•  {item}", cx+0.15, cy+0.7+j*0.72, cw-0.25, 0.68,
                    font_size=11, color=DARK)

add_footer_line(slide)
add_slide_number(slide, 11)

# ═══════════════════════════════════════════════════════════════════════════════
# Slide 12 — Conclusion
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK_LAYOUT)
add_crimson_bar(slide)
slide_heading(slide, "Conclusion")

built = [
    "Dual YOLOv11 pipeline: spots.pt (bay detection) + best.pt (vehicle tracking)",
    "Bird's Eye View perspective normalisation eliminates camera distortion",
    "Greedy BEV matching assigns cars to spots in O(K log K) per frame",
    "Self-healing recalibration at 2 AM — zero manual re-annotation",
    "Deployed model (y11s): mAP@50 = 96.4%, mAP@50-95 = 80.7%",
]
add_textbox(slide, "What We Built", 0.5, 1.2, 7.5, 0.42,
            font_size=14, bold=True, color=DARK)
for i, b in enumerate(built):
    add_textbox(slide, f"• {b}", 0.65, 1.65+i*0.62, 7.2, 0.58,
                font_size=12, color=DARK)

# Matters box
add_rect(slide, 0.4, 4.9, 7.5, 1.6, fill_rgb=LIGHTGREY)
add_rect(slide, 0.4, 4.9, 0.06, 1.6, fill_rgb=CRIMSON)
add_textbox(slide, "Why It Matters", 0.6, 4.97, 7.0, 0.42, font_size=13, bold=True, color=CRIMSON)
add_textbox(slide,
    "High-accuracy real-time parking management with no per-bay hardware.\n"
    "Purely software-based — cost-effective, scalable, and maintainable.",
    0.6, 5.4, 7.0, 0.95, font_size=12, color=DARK)

# Next steps panel
add_rect(slide, 8.15, 1.2, 5.0, 5.3, fill_rgb=LIGHTGREY)
add_rect(slide, 8.15, 1.2, 5.0, 0.06, fill_rgb=CRIMSON)
add_textbox(slide, "Next Steps", 8.3, 1.3, 4.7, 0.42, font_size=14, bold=True, color=CRIMSON)
nexts = [
    "Multi-camera facility-wide coverage",
    "IoT / MQTT real-time data streaming",
    "Mobile app driver notifications",
    "ALPR for permit enforcement",
    "Edge deployment (Jetson / Coral TPU)",
    "Adaptive BEV threshold learning",
]
for i, n in enumerate(nexts):
    add_textbox(slide, f"→  {n}", 8.3, 1.85+i*0.72, 4.7, 0.65,
                font_size=12, color=DARK)

# Bottom crimson thank you bar
add_rect(slide, 0, 6.8, 13.33, 0.7, fill_rgb=CRIMSON)
add_textbox(slide, "Thank You  |  Q & A",
            0, 6.83, 13.33, 0.55,
            font_size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

add_footer_line(slide)
add_slide_number(slide, 12)

# ═══════════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════════
OUT = "/Users/jul/Desktop/uni/parking-vision-system/documents/Parking_Vision_System_Presentation.pptx"
prs.save(OUT)
print(f"✅  Saved: {OUT}")
