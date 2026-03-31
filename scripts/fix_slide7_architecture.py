"""
Fix slide 7 system architecture diagram:

1. Move NMS Filter from Main Processing Loop → Initialization Phase
   New flow: Reference.jpg → spots.pt → NMS Filter → spots_data.json

2. Delete the arrow from NMS Filter → best.pt  (Shape 61)

3. Delete dotted lines from config.yaml → BEV and homography.json → BEV
   (Shape 62, Shape 63)

4. Delete the false "Self-Healing Recalibration at 2:00 AM" loop label
   under config.yaml / homography.json  (Text 40)

Input:  Parking_Presentation_Final_Fixed.pptx
Output: Parking_Presentation_Final_Fixed.pptx  (overwrite in-place)
"""
import copy
import os
from lxml import etree
from pptx import Presentation
from pptx.util import Inches, Emu

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PPTX = os.path.join(BASE_DIR, "Parking_Presentation_Final_Fixed.pptx")

# EMU helpers
def emu(inches): return int(inches * 914400)

def set_xfrm(shape, left_in, top_in, w_in, h_in):
    """Update a shape's position and size."""
    shape.left   = emu(left_in)
    shape.top    = emu(top_in)
    shape.width  = emu(w_in)
    shape.height = emu(h_in)

def find(slide, name):
    for s in slide.shapes:
        if s.name == name:
            return s
    return None

def delete_shape(slide, name):
    s = find(slide, name)
    if s:
        s._element.getparent().remove(s._element)
        print(f"  Deleted: {name}")
    else:
        print(f"  WARNING: {name} not found")

def clone_arrow(slide, src_name, left_in, top_in, w_in, h_in, new_name):
    """Clone an existing line-arrow shape and reposition it."""
    src = find(slide, src_name)
    if src is None:
        print(f"  WARNING: clone source {src_name!r} not found")
        return
    new_elem = copy.deepcopy(src._element)
    # Update id and name
    cNvPr = new_elem.find('.//{http://schemas.openxmlformats.org/presentationml/2006/main}cNvPr')
    if cNvPr is None:
        cNvPr = new_elem.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}cNvPr')
    # Give a unique id — just use a large number unlikely to collide
    existing_ids = {int(e.get('id')) for e in slide._element.iter()
                    if e.get('id') is not None and e.get('id').isdigit()}
    new_id = max(existing_ids, default=100) + 1
    for pr in new_elem.iter():
        if pr.get('id') is not None:
            pr.set('id', str(new_id))
            pr.set('name', new_name)
            break
    # Update xfrm
    a_ns = 'http://schemas.openxmlformats.org/drawingml/2006/main'
    off = new_elem.find(f'.//{{{a_ns}}}off')
    ext = new_elem.find(f'.//{{{a_ns}}}ext')
    off.set('x', str(emu(left_in)))
    off.set('y', str(emu(top_in)))
    ext.set('cx', str(emu(w_in)))
    ext.set('cy', str(emu(h_in)))
    slide.shapes._spTree.append(new_elem)
    print(f"  Added arrow: {new_name} at ({left_in:.2f}\", {top_in:.2f}\")")


def fix_slide7(prs):
    slide = prs.slides[6]   # 0-indexed → slide 7
    print("=== Fixing slide 7 ===")

    # ── STEP 1: Delete bad shapes ──────────────────────────────────────────
    print("\n[ Deleting ]")
    delete_shape(slide, "Text 40")    # false recal loop label
    delete_shape(slide, "Shape 61")   # NMS → best.pt arrow
    delete_shape(slide, "Shape 62")   # config.yaml → BEV arrow
    delete_shape(slide, "Shape 63")   # homography.json → BEV arrow

    # ── STEP 2: Move NMS Filter box into Initialization Phase ──────────────
    # New position: directly below spots.pt (same left/width), at y=3.00"
    # spots.pt is at left=2.22" w=1.65" h=0.80"  →  bottom=2.92"
    # NMS will be at y=3.00" (small gap below spots.pt), same x and width
    print("\n[ Moving NMS Filter ]")
    NMS_LEFT, NMS_TOP, NMS_W, NMS_H = 2.22, 3.00, 1.65, 0.65
    nms_bg  = find(slide, "Shape 34")
    nms_txt = find(slide, "Text 35")
    if nms_bg:
        set_xfrm(nms_bg,  NMS_LEFT, NMS_TOP, NMS_W, NMS_H)
        print(f"  Moved Shape 34 (NMS bg) → ({NMS_LEFT}\", {NMS_TOP}\")")
    if nms_txt:
        set_xfrm(nms_txt, NMS_LEFT, NMS_TOP, NMS_W, NMS_H)
        print(f"  Moved Text 35  (NMS txt)→ ({NMS_LEFT}\", {NMS_TOP}\")")

    # ── STEP 3: Move spots_data.json below NMS ─────────────────────────────
    # NMS bottom = 3.00 + 0.65 = 3.65"  →  spots_data starts at 3.83"
    print("\n[ Moving spots_data.json ]")
    SD_LEFT, SD_TOP, SD_W, SD_H = 2.22, 3.83, 1.65, 0.72
    sd_bg  = find(slide, "Shape 23")
    sd_txt = find(slide, "Text 24")
    if sd_bg:
        set_xfrm(sd_bg,  SD_LEFT, SD_TOP, SD_W, SD_H)
        print(f"  Moved Shape 23 (json bg) → ({SD_LEFT}\", {SD_TOP}\")")
    if sd_txt:
        set_xfrm(sd_txt, SD_LEFT, SD_TOP, SD_W, SD_H)
        print(f"  Moved Text 24  (json txt)→ ({SD_LEFT}\", {SD_TOP}\")")

    # ── STEP 4: Reposition existing vertical arrow (spots.pt → NMS) ────────
    # spots.pt bottom-center: x=2.22+0.825=3.047", y=2.92"
    # NMS top: y=3.00"  →  gap=0.08"
    # Use a near-vertical line (tiny cx, gap cy)
    print("\n[ Adjusting spots.pt → NMS arrow ]")
    arr = find(slide, "Shape 21")
    if arr:
        # left positioned near center of spots.pt (3.047" - 0.01")
        set_xfrm(arr, 3.027, 2.92, 0.02, 0.08)
        print("  Repositioned Shape 21 (spots.pt→NMS)")

    # ── STEP 5: Add new arrow NMS → spots_data.json ────────────────────────
    # NMS bottom = 3.65",  spots_data top = 3.83"  → gap = 0.18"
    print("\n[ Adding NMS → spots_data arrow ]")
    clone_arrow(slide, "Shape 21",
                left_in=3.027, top_in=3.65, w_in=0.02, h_in=0.18,
                new_name="Arrow NMS-to-JSON")

    # ── STEP 6: Expand Initialization Phase box to contain new layout ──────
    # Current box [13]: left=0.22" top=2.00" w=4.00" h=2.78" → bottom=4.78"
    # New content bottom: spots_data bottom = 3.83+0.72=4.55"  → fits, no change needed
    # But adjust width to fully contain spots.pt/NMS/JSON right edge=2.22+1.65=3.87"
    # Current box right=4.22" > 3.87" → already fits, no change needed
    print("\n[ Initialization Phase box already fits — no resize needed ]")
    init_box = find(slide, "Shape 13")
    if init_box:
        print(f"  Shape 13 (init box): w={init_box.width/914400:.2f}\" right={(init_box.left+init_box.width)/914400:.2f}\"")

    print("\nDone fixing slide 7.")


def main():
    prs = Presentation(PPTX)
    fix_slide7(prs)
    prs.save(PPTX)
    print(f"\nSaved: {PPTX}")


if __name__ == "__main__":
    main()
