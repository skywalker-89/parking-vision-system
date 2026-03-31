import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_IMG = os.path.join(BASE_DIR, "config", "reference.jpg")
SPOTS_JSON = os.path.join(BASE_DIR, "config", "spots_data.json")
HOMOGRAPHY_JSON = os.path.join(BASE_DIR, "config", "homography.json")

# Load the actual calibration from homography.json (source of truth at runtime)
with open(HOMOGRAPHY_JSON) as _f:
    _hdata = json.load(_f)
SRC_POINTS = np.float32(_hdata["src_points"])
DST_POINTS = np.float32(_hdata["dst_points"])
BEV_W = _hdata["birdseye_w"]
BEV_H = _hdata["birdseye_h"]
HOMOGRAPHY_MATRIX = np.float32(_hdata["matrix"])

DIST_THRESH = 70

# The homography matrix was calibrated at 1920x1080 (see perspective.py defaults).
# reference.jpg and spots_data.json use the native camera resolution (2688x1520).
# We must resize to calibration resolution before warping, and scale spot coords too.
CAL_W, CAL_H = 1920, 1080


def get_perspective_matrix():
    return HOMOGRAPHY_MATRIX


def transform_point(x, y, matrix):
    pt = np.array([[[x, y]]], dtype="float32")
    return cv2.perspectiveTransform(pt, matrix)[0][0]


def load_spots():
    try:
        with open(SPOTS_JSON) as f:
            return json.load(f).get("spots", [])
    except FileNotFoundError:
        return []


def ref_to_cal(img):
    """Resize reference image to calibration resolution for correct warpPerspective."""
    return cv2.resize(img, (CAL_W, CAL_H))


def scale_spot_to_cal(spot, ref_w, ref_h):
    """Scale a spot bbox from native ref resolution to calibration resolution."""
    x1, y1, x2, y2 = spot
    sx, sy = CAL_W / ref_w, CAL_H / ref_h
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def draw_label(img, text, pos, font_scale=0.9, thickness=2, text_color=(255, 255, 0), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + 6), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────
# IMAGE 1: Homography Calibration
# ─────────────────────────────────────────────
def gen_homography_img():
    ref = cv2.imread(REF_IMG)
    if ref is None:
        print("Reference image not found!")
        return

    ref_h, ref_w = ref.shape[:2]
    PANEL_H = 700   # fixed output height for both panels

    # ── Left panel: camera view annotated ──
    # SRC_POINTS are in CAL (1920x1080) space; scale them to display size
    panel_left_w = int(CAL_W * PANEL_H / CAL_H)
    left = cv2.resize(ref_to_cal(ref), (panel_left_w, PANEL_H))
    scale = PANEL_H / CAL_H

    src_scaled = (SRC_POINTS * scale).astype(np.int32)

    # Semi-transparent fill inside trapezoid
    overlay = left.copy()
    cv2.fillPoly(overlay, [src_scaled], (0, 0, 180))
    cv2.addWeighted(overlay, 0.30, left, 0.70, 0, left)
    cv2.polylines(left, [src_scaled], isClosed=True, color=(0, 0, 255), thickness=4)

    # Draw numbered corner points with coordinate labels
    corner_labels = [f"P{i} ({int(p[0])},{int(p[1])})" for i, p in enumerate(SRC_POINTS)]
    # Offset label away from edge: push inward for corners near image border
    label_offsets = [(8, 28), (-160, 28), (-190, -18), (8, -18)]
    for pt, label, off in zip(src_scaled, corner_labels, label_offsets):
        cv2.circle(left, tuple(pt), 14, (0, 255, 255), -1)
        cv2.circle(left, tuple(pt), 14, (0, 0, 0), 2)
        lx = max(4, min(pt[0] + off[0], panel_left_w - 200))
        ly = max(24, min(pt[1] + off[1], PANEL_H - 8))
        draw_label(left, label, (lx, ly), font_scale=0.72, thickness=2,
                   text_color=(0, 255, 255), bg_color=(20, 20, 20))

    draw_label(left, "Camera View  (src_points)", (14, 36),
               font_scale=0.95, thickness=2, text_color=(255, 255, 255), bg_color=(20, 20, 20))

    # ── Right panel: actual warped BEV image ──
    # Must warp at calibration resolution (1920x1080) - that's what the matrix was computed for
    bev_raw = cv2.warpPerspective(ref_to_cal(ref), get_perspective_matrix(), (BEV_W, BEV_H))

    # Scale BEV to fill panel height, keep aspect ratio, then pad to match left width
    bev_disp_h = PANEL_H
    bev_disp_w = int(BEV_W * bev_disp_h / BEV_H)
    bev_scaled = cv2.resize(bev_raw, (bev_disp_w, bev_disp_h))

    # Pad horizontally to match left panel width
    right = np.ones((PANEL_H, panel_left_w, 3), dtype=np.uint8) * 40
    x_off = (panel_left_w - bev_disp_w) // 2
    right[:, x_off: x_off + bev_disp_w] = bev_scaled

    # Mark the 4 dst corners on the BEV panel
    dst_corners_img = [(x_off, 0), (x_off + bev_disp_w - 1, 0),
                       (x_off + bev_disp_w - 1, PANEL_H - 1), (x_off, PANEL_H - 1)]
    dst_labels = [f"({int(p[0])},{int(p[1])})" for p in DST_POINTS]
    corner_label_offsets = [(6, 26), (-90, 26), (-90, -12), (6, -12)]
    for (px, py), label, off in zip(dst_corners_img, dst_labels, corner_label_offsets):
        cv2.circle(right, (px, py), 14, (0, 255, 255), -1)
        cv2.circle(right, (px, py), 14, (0, 0, 0), 2)
        draw_label(right, label, (px + off[0], py + off[1]),
                   font_scale=0.72, thickness=2,
                   text_color=(0, 255, 255), bg_color=(20, 20, 20))

    draw_label(right, "BEV Warped Output  (dst_points)", (14, 36),
               font_scale=0.95, thickness=2, text_color=(255, 255, 255), bg_color=(20, 20, 20))

    # ── Center arrow strip ──
    arrow_strip = np.ones((PANEL_H, 130, 3), dtype=np.uint8) * 18
    mid_y = PANEL_H // 2
    cv2.arrowedLine(arrow_strip, (12, mid_y), (110, mid_y), (0, 200, 255), 4, tipLength=0.2)
    draw_label(arrow_strip, "H", (50, mid_y - 18), font_scale=1.1, thickness=2,
               text_color=(0, 200, 255), bg_color=(18, 18, 18))
    draw_label(arrow_strip, "getPerspective", (5, mid_y + 22), font_scale=0.40, thickness=1,
               text_color=(160, 160, 160), bg_color=(18, 18, 18))
    draw_label(arrow_strip, "Transform(src,dst)", (5, mid_y + 38), font_scale=0.40, thickness=1,
               text_color=(160, 160, 160), bg_color=(18, 18, 18))

    # ── Compose ──
    combined = np.concatenate([left, arrow_strip, right], axis=1)
    title_bar = np.ones((60, combined.shape[1], 3), dtype=np.uint8) * 20
    draw_label(title_bar, "Homography Calibration", (combined.shape[1] // 2 - 230, 42),
               font_scale=1.3, thickness=3, text_color=(255, 255, 255), bg_color=(20, 20, 20))
    out_img = np.vstack([title_bar, combined])

    out = os.path.join(BASE_DIR, "homography_calibration.jpg")
    cv2.imwrite(out, out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved {out}")


# ─────────────────────────────────────────────
# IMAGE 2: BEV Projection
# ─────────────────────────────────────────────
def gen_bev_img():
    ref = cv2.imread(REF_IMG)
    if ref is None:
        print("Reference image not found!")
        return

    matrix = get_perspective_matrix()
    spots = load_spots()
    ref_h, ref_w = ref.shape[:2]
    PANEL_H = 700

    # Display at calibration resolution so spot coords (scaled to cal) line up correctly
    panel_cam_w = int(CAL_W * PANEL_H / CAL_H)
    left = cv2.resize(ref_to_cal(ref), (panel_cam_w, PANEL_H))
    disp_scale = PANEL_H / CAL_H   # from CAL (1920x1080) → display

    np.random.seed(7)
    sim_indices = set(np.random.choice(len(spots), min(9, len(spots)), replace=False).tolist())
    car_spots_cal = []   # spots in cal resolution coords

    for i, spot in enumerate(spots):
        # Scale from native ref resolution → cal resolution → display
        cs = scale_spot_to_cal(spot, ref_w, ref_h)
        x1d, y1d, x2d, y2d = [int(c * disp_scale) for c in cs]
        cv2.rectangle(left, (x1d, y1d), (x2d, y2d), (0, 230, 0), 2)
        if i in sim_indices:
            cx, cy = (x1d + x2d) // 2, y2d - 6
            cw = int((x2d - x1d) * 0.80)
            ch = int((y2d - y1d) * 0.75)
            cv2.rectangle(left, (cx - cw // 2, cy - ch), (cx + cw // 2, cy), (30, 30, 220), -1)
            cv2.rectangle(left, (cx - cw // 2, cy - ch), (cx + cw // 2, cy), (80, 80, 255), 2)
            car_spots_cal.append(scale_spot_to_cal(spot, ref_w, ref_h))

    draw_label(left, "Camera View", (14, 36), font_scale=0.95, thickness=2,
               text_color=(255, 255, 255), bg_color=(20, 20, 20))
    cv2.rectangle(left, (14, PANEL_H - 68), (40, PANEL_H - 50), (0, 230, 0), 2)
    draw_label(left, "Parking Spot", (46, PANEL_H - 50), font_scale=0.62, thickness=1,
               text_color=(0, 230, 0), bg_color=(20, 20, 20))
    cv2.rectangle(left, (14, PANEL_H - 38), (40, PANEL_H - 20), (30, 30, 220), -1)
    draw_label(left, "Simulated Vehicle", (46, PANEL_H - 20), font_scale=0.62, thickness=1,
               text_color=(100, 120, 255), bg_color=(20, 20, 20))

    # ── Right panel: actual warped BEV (warp at calibration res) ──
    bev_raw = cv2.warpPerspective(ref_to_cal(ref), matrix, (BEV_W, BEV_H))

    bev_disp_h = PANEL_H
    bev_disp_w = int(BEV_W * bev_disp_h / BEV_H)
    bev_scaled = cv2.resize(bev_raw, (bev_disp_w, bev_disp_h))

    right = np.ones((PANEL_H, panel_cam_w, 3), dtype=np.uint8) * 28
    x_off = (panel_cam_w - bev_disp_w) // 2
    right[:, x_off: x_off + bev_disp_w] = bev_scaled

    # Overlay spots: transform cal-resolution spot centers through matrix
    for spot in spots:
        cs = scale_spot_to_cal(spot, ref_w, ref_h)
        cx, cy = (cs[0] + cs[2]) / 2, (cs[1] + cs[3]) / 2
        bx, by = transform_point(cx, cy, matrix)
        if not (0 <= bx <= BEV_W and 0 <= by <= BEV_H):
            continue
        bx_draw = int(x_off + bx * bev_disp_w / BEV_W)
        by_draw = int(by * bev_disp_h / BEV_H)
        rw = max(6, int(20 * bev_disp_w / BEV_W))
        rh = max(10, int(40 * bev_disp_h / BEV_H))
        cv2.rectangle(right, (bx_draw - rw, by_draw - rh), (bx_draw + rw, by_draw + rh), (0, 255, 0), 2)

    # Overlay simulated car dots in BEV
    for cs in car_spots_cal:
        bx, by = transform_point((cs[0] + cs[2]) / 2, cs[3], matrix)
        if not (0 <= bx <= BEV_W and 0 <= by <= BEV_H):
            continue
        bx_draw = int(x_off + bx * bev_disp_w / BEV_W)
        by_draw = int(by * bev_disp_h / BEV_H)
        cv2.circle(right, (bx_draw, by_draw), 10, (0, 0, 255), -1)
        cv2.circle(right, (bx_draw, by_draw), 10, (80, 80, 255), 2)

    draw_label(right, "Bird's Eye View (BEV)", (14, 36), font_scale=0.95, thickness=2,
               text_color=(255, 255, 255), bg_color=(20, 20, 20))
    cv2.rectangle(right, (14, PANEL_H - 68), (40, PANEL_H - 50), (0, 255, 0), 2)
    draw_label(right, "Parking Spot BEV", (46, PANEL_H - 50), font_scale=0.62, thickness=1,
               text_color=(0, 230, 0), bg_color=(20, 20, 20))
    cv2.circle(right, (27, PANEL_H - 32), 8, (0, 0, 255), -1)
    draw_label(right, "Car Tire Footprint", (46, PANEL_H - 20), font_scale=0.62, thickness=1,
               text_color=(100, 120, 255), bg_color=(20, 20, 20))

    # ── Center arrow strip ──
    arrow_strip = np.ones((PANEL_H, 130, 3), dtype=np.uint8) * 18
    mid_y = PANEL_H // 2
    cv2.arrowedLine(arrow_strip, (12, mid_y), (110, mid_y), (0, 200, 255), 4, tipLength=0.2)
    draw_label(arrow_strip, "H*p", (44, mid_y - 18), font_scale=0.9, thickness=2,
               text_color=(0, 200, 255), bg_color=(18, 18, 18))
    draw_label(arrow_strip, "Perspective", (8, mid_y + 22), font_scale=0.40, thickness=1,
               text_color=(160, 160, 160), bg_color=(18, 18, 18))
    draw_label(arrow_strip, "Warp", (30, mid_y + 38), font_scale=0.40, thickness=1,
               text_color=(160, 160, 160), bg_color=(18, 18, 18))

    combined = np.concatenate([left, arrow_strip, right], axis=1)
    title_bar = np.ones((60, combined.shape[1], 3), dtype=np.uint8) * 20
    draw_label(title_bar, "BEV Projection", (combined.shape[1] // 2 - 165, 42),
               font_scale=1.3, thickness=3, text_color=(255, 255, 255), bg_color=(20, 20, 20))
    out_img = np.vstack([title_bar, combined])

    out = os.path.join(BASE_DIR, "bev_projection.jpg")
    cv2.imwrite(out, out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved {out}")


# ─────────────────────────────────────────────
# IMAGE 3: Greedy Matching O(K log K)
# ─────────────────────────────────────────────
def gen_greedy_img():
    matrix = get_perspective_matrix()
    spots = load_spots()

    # Need native ref resolution to scale spots to calibration space
    ref = cv2.imread(REF_IMG)
    ref_h, ref_w = (ref.shape[:2] if ref is not None else (1520, 2688))

    spot_bev = []
    for spot in spots[14:27]:
        cs = scale_spot_to_cal(spot, ref_w, ref_h)
        bx, by = transform_point((cs[0] + cs[2]) / 2, (cs[1] + cs[3]) / 2, matrix)
        if 0 <= bx <= BEV_W and 0 <= by <= BEV_H:
            spot_bev.append([bx, by])

    if not spot_bev:
        spot_bev = [[100 + i * 40, 200 + (i % 3) * 80] for i in range(8)]

    spot_bev = np.array(spot_bev)

    np.random.seed(42)
    # Most cars near spots, a couple unmatched
    car_indices = np.random.choice(len(spot_bev), size=9, replace=False)
    cars_bev = spot_bev[car_indices] + np.random.normal(0, 28, (9, 2))
    # Add 2 unmatched noise cars well outside any spot
    noise_cars = np.array([[spot_bev[:, 0].mean() + 120, spot_bev[:, 1].mean() - 80],
                           [spot_bev[:, 0].min() - 90,   spot_bev[:, 1].mean() + 60]])
    cars_bev = np.vstack([cars_bev, noise_cars])

    # ── Run greedy matching ──
    candidates = []
    for ci, (cx, cy) in enumerate(cars_bev):
        for si, (sx, sy) in enumerate(spot_bev):
            d = np.hypot(cx - sx, cy - sy)
            if d < DIST_THRESH:
                candidates.append((d, ci, si))
    candidates.sort(key=lambda x: x[0])

    matched_cars, matched_spots = set(), set()
    matches = []
    for d, ci, si in candidates:
        if ci not in matched_cars and si not in matched_spots:
            matches.append((ci, si, d))
            matched_cars.add(ci)
            matched_spots.add(si)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(13, 9), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.invert_yaxis()

    # Step 1: faint lines for all N×M pairs (out of range)
    for ci, (cx, cy) in enumerate(cars_bev):
        for si, (sx, sy) in enumerate(spot_bev):
            d = np.hypot(cx - sx, cy - sy)
            if d >= DIST_THRESH:
                ax.plot([cx, sx], [cy, sy], color="#555577", linestyle=":", linewidth=0.7, alpha=0.35, zorder=1)

    # Step 2: valid candidate lines (in range, not yet matched)
    for d, ci, si in candidates:
        if not any(ci == m[0] and si == m[1] for m in matches):
            cx, cy = cars_bev[ci]
            sx, sy = spot_bev[si]
            ax.plot([cx, sx], [cy, sy], color="#03A9F4", linestyle="--", linewidth=1.6, alpha=0.55, zorder=2)

    # Step 3: matched pairs — thick green lines
    for ci, si, d in matches:
        cx, cy = cars_bev[ci]
        sx, sy = spot_bev[si]
        ax.plot([cx, sx], [cy, sy], color="#00E676", linestyle="-", linewidth=3.0, zorder=4)
        mx, my = (cx + sx) / 2, (cy + sy) / 2
        ax.text(mx + 3, my - 4, f"{int(d)}px", fontsize=8, color="#B9F6CA",
                fontweight="bold", zorder=6)
        ax.text(mx + 3, my + 10, "✓", fontsize=11, color="#00E676", zorder=6)

    # Threshold circles per spot
    for sx, sy in spot_bev:
        circle = plt.Circle((sx, sy), DIST_THRESH, color="#4CAF50",
                             fill=False, linestyle="--", alpha=0.4, linewidth=1.5)
        ax.add_patch(circle)

    # Draw spots
    ax.scatter(spot_bev[:, 0], spot_bev[:, 1],
               c="#4CAF50", marker="s", s=320, zorder=5,
               edgecolors="#A5D6A7", linewidths=1.5, label="Parking Spot")

    # Draw matched cars (filled red)
    for ci in matched_cars:
        cx, cy = cars_bev[ci]
        ax.scatter(cx, cy, c="#FF5252", marker="o", s=160, zorder=5,
                   edgecolors="#FF8A80", linewidths=1.5)

    # Draw unmatched cars (hollow red)
    unmatched_car_indices = set(range(len(cars_bev))) - matched_cars
    for ci in unmatched_car_indices:
        cx, cy = cars_bev[ci]
        ax.scatter(cx, cy, facecolors="none", edgecolors="#FF5252",
                   marker="o", s=180, linewidths=2.5, zorder=5)
        ax.text(cx + 5, cy - 12, "unmatched", fontsize=8, color="#FF8A80")

    # ── Algorithm steps annotation ──
    steps_text = (
        "Algorithm Steps:\n"
        "① Compute all N×M distances\n"
        "② Filter candidates: dist < 70px  →  K pairs\n"
        "③ Sort K candidates  →  O(K log K)\n"
        "④ Greedy assign: iterate sorted list,\n"
        "   assign if both car & spot free"
    )
    ax.text(0.985, 0.03, steps_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", horizontalalignment="right",
            color="#CFD8DC",
            bbox=dict(boxstyle="round,pad=0.6", fc="#0d0d1a", ec="#546E7A", lw=1.5))

    # ── Legend ──
    legend_elements = [
        mpatches.Patch(facecolor="#4CAF50", edgecolor="#A5D6A7", label="Parking Spot"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF5252",
               markeredgecolor="#FF8A80", markersize=10, label="Car (matched)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="#FF5252", markersize=10, markeredgewidth=2, label="Car (unmatched)"),
        Line2D([0], [0], color="#00E676", linewidth=2.5, label="Matched pair (greedy assign)"),
        Line2D([0], [0], color="#03A9F4", linewidth=1.5, linestyle="--", label="Candidate pair (in range)"),
        Line2D([0], [0], color="#555577", linewidth=1.0, linestyle=":", label="Out-of-range pair"),
        Line2D([0], [0], color="#4CAF50", linewidth=1.5, linestyle="--",
               alpha=0.5, label=f"{DIST_THRESH}px match threshold"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              fontsize=9, frameon=True,
              facecolor="#0d0d1a", edgecolor="#546E7A", labelcolor="#CFD8DC")

    ax.set_title("Greedy BEV Matching   O(K log K)",
                 fontsize=17, fontweight="bold", color="#ECEFF1", pad=16)
    ax.tick_params(colors="#607D8B")
    for spine in ax.spines.values():
        spine.set_edgecolor("#37474F")
    ax.grid(True, linestyle="--", alpha=0.2, color="#546E7A")

    out = os.path.join(BASE_DIR, "greedy_matching.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────
# IMAGE 4: Parking Spot Detection - Reference Image
# ─────────────────────────────────────────────
def gen_spot_detection_img():
    ref = cv2.imread(REF_IMG)
    if ref is None:
        print("Reference image not found!")
        return

    spots = load_spots()
    if not spots:
        print("No spots found in spots_data.json!")
        return

    ref_h, ref_w = ref.shape[:2]

    # Work at full native resolution for sharpness, then downscale for output
    img = ref.copy()

    FREE_COLOR   = (0, 220, 0)     # green — all free (reference has no cars)
    TEXT_COLOR   = (255, 255, 255)
    BG_COLOR     = (20, 20, 20)
    BORDER_COLOR = (0, 180, 0)

    for i, spot in enumerate(spots):
        x1, y1, x2, y2 = map(int, spot)

        # Semi-transparent green fill
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), FREE_COLOR, -1)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

        # Border
        cv2.rectangle(img, (x1, y1), (x2, y2), FREE_COLOR, 3)

        # Spot number label inside the box
        label = str(i + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = max(0.5, min(0.9, (x2 - x1) / 80))
        (tw, th), _ = cv2.getTextSize(label, font, fs, 2)
        lx = x1 + ((x2 - x1) - tw) // 2
        ly = y1 + ((y2 - y1) + th) // 2
        cv2.rectangle(img, (lx - 3, ly - th - 4), (lx + tw + 3, ly + 4), BG_COLOR, -1)
        cv2.putText(img, label, (lx, ly), font, fs, TEXT_COLOR, 2, cv2.LINE_AA)

    # Title bar overlay at top
    cv2.rectangle(img, (0, 0), (ref_w, 70), (20, 20, 20), -1)
    draw_label(img, "Parking Spot Detection  -  Reference Image",
               (ref_w // 2 - 480, 48),
               font_scale=1.6, thickness=3,
               text_color=(255, 255, 255), bg_color=(20, 20, 20))

    # Stats badge top-right
    badge = f"{len(spots)} spots detected  |  spots.pt (YOLOv11)"
    draw_label(img, badge, (ref_w - 680, 48),
               font_scale=0.85, thickness=2,
               text_color=(0, 230, 0), bg_color=(20, 20, 20))

    # Legend bottom-left
    lx0, ly0 = 20, ref_h - 60
    cv2.rectangle(img, (lx0, ly0), (lx0 + 34, ly0 + 28), FREE_COLOR, -1)
    cv2.rectangle(img, (lx0, ly0), (lx0 + 34, ly0 + 28), BORDER_COLOR, 2)
    draw_label(img, "FREE  (no vehicles in reference)", (lx0 + 42, ly0 + 24),
               font_scale=0.8, thickness=2,
               text_color=(0, 220, 0), bg_color=(20, 20, 20))

    # Downscale to 1920-wide for reasonable file size
    out_w = 1920
    out_h = int(ref_h * out_w / ref_w)
    img_out = cv2.resize(img, (out_w, out_h))

    out = os.path.join(BASE_DIR, "spot_detection_reference.jpg")
    cv2.imwrite(out, img_out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved {out}")


if __name__ == "__main__":
    gen_homography_img()
    gen_bev_img()
    gen_greedy_img()
    gen_spot_detection_img()
