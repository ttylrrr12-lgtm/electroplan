
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Literal, Optional, Tuple, Dict
from pydantic import BaseModel
import numpy as np
import cv2
from PIL import Image
import io, base64

from .seg_config import load_config

router = APIRouter(prefix="/api", tags=["segmentation-v2"])

# ---------- helpers ----------

def _img_to_dataurl(img):
    if len(img.shape) == 2:
        mode = "L"; arr = img
    else:
        mode = "RGB"; arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def _resize_max_side(img, max_side):
    h, w = img.shape[:2]; s = max(h, w)
    if s <= max_side: return img, 1.0
    scale = max_side / float(s)
    out = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return out, scale

def _binarize(img, method, block_size, C):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    if method == "otsu":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block_size, C)
    return bw

def _deskew(bw):
    edges = cv2.Canny(bw, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is None: return bw
    angles = []
    for rho, theta in lines[:,0,:]:
        a = (theta * 180 / np.pi) % 180
        if a > 135: a -= 180
        angles.append(a)
    if not angles: return bw
    median = float(np.median(angles))
    target = min([ -90, -45, 0, 45, 90 ], key=lambda t: abs(median - t))
    delta = median - target
    if abs(delta) < 0.5: return bw
    h, w = bw.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), delta, 1.0)
    return cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

def _remove_text(bw, kernel=3, dilate=2, inpaint_radius=2):
    # thin strokes ~ text: use morphological tophat-ish via opening on inverse
    inv = 255 - bw
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    small = cv2.morphologyEx(inv, cv2.MORPH_OPEN, k)
    mask = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate)), iterations=1)
    # Inpaint on original grayscale background of inverse
    base = inv.copy()
    cleaned = cv2.inpaint(base, mask, inpaint_radius, cv2.INPAINT_TELEA)
    out = 255 - cleaned
    return out, mask

def _extract_lines(bw, threshold, min_len_px, max_gap_px):
    edges = cv2.Canny(bw, 60, 160, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold,
                            minLineLength=min_len_px, maxLineGap=max_gap_px)
    segments = []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            segments.append((float(x1),float(y1),float(x2),float(y2)))
    return segments, edges

def _snap_ortho(segments, angle_tol_deg=10):
    snapped = []
    for x1,y1,x2,y2 in segments:
        dx, dy = x2-x1, y2-y1
        ang = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0
        if min(abs(ang-0), abs(ang-180)) < angle_tol_deg:
            ym = (y1 + y2)/2.0; snapped.append((x1, ym, x2, ym))
        elif min(abs(ang-90), abs(ang-270)) < angle_tol_deg or abs(ang-90) < angle_tol_deg:
            xm = (x1 + x2)/2.0; snapped.append((xm, y1, xm, y2))
        else:
            snapped.append((x1,y1,x2,y2))
    return snapped

def _grid_snap(v, step): return round(v/step)*step

def _merge_collinear(segments, gap_tol_px=15, min_seg_px=24):
    horiz = {}; vert = {}
    for x1,y1,x2,y2 in segments:
        if abs(y1-y2) < 1.0:
            y = round((y1+y2)/2); a,b = sorted([x1,x2]); horiz.setdefault(y, []).append([a,b])
        elif abs(x1-x2) < 1.0:
            x = round((x1+x2)/2); a,b = sorted([y1,y2]); vert.setdefault(x, []).append([a,b])
    merged = []
    for y, spans in horiz.items():
        spans.sort(); cur = list(spans[0])
        for a,b in spans[1:]:
            if a - cur[1] <= gap_tol_px: cur[1] = max(cur[1], b)
            else: merged.append((cur[0], y, cur[1], y)); cur = [a,b]
        merged.append((cur[0], y, cur[1], y))
    for x, spans in vert.items():
        spans.sort(); cur = list(spans[0])
        for a,b in spans[1:]:
            if a - cur[1] <= gap_tol_px: cur[1] = max(cur[1], b)
            else: merged.append((x, cur[0], x, cur[1])); cur = [a,b]
        merged.append((x, cur[0], x, cur[1]))
    out = []
    for x1,y1,x2,y2 in merged:
        if ((x2-x1)**2 + (y2-y1)**2) >= (min_seg_px*min_seg_px):
            out.append((float(x1),float(y1),float(x2),float(y2)))
    return out

def _detect_openings_doors(merged_segments, min_gap_px=40, max_gap_px=200):
    doors = []; horiz_by_y={}; vert_by_x={}
    for x1,y1,x2,y2 in merged_segments:
        if abs(y1-y2) < 1.0:
            y = y1; a,b = sorted([x1,x2]); horiz_by_y.setdefault(y, []).append((a,b))
        elif abs(x1-x2) < 1.0:
            x = x1; a,b = sorted([y1,y2]); vert_by_x.setdefault(x, []).append((a,b))
    for y, spans in horiz_by_y.items():
        spans.sort()
        for i in range(len(spans)-1):
            a1,b1 = spans[i]; a2,b2 = spans[i+1]
            gap = a2 - b1
            if min_gap_px <= gap <= max_gap_px: doors.append(("door", b1, y, a2, y))
    for x, spans in vert_by_x.items():
        spans.sort()
        for i in range(len(spans)-1):
            a1,b1 = spans[i]; a2,b2 = spans[i+1]
            gap = a2 - b1
            if min_gap_px <= gap <= max_gap_px: doors.append(("door", x, b1, x, a2))
    return doors

def _detect_windows_from_pairs(merged_segments, min_w_px, max_w_px, delta_pair_px=6):
    # Pair parallel segments separated by window width
    windows = []
    horiz = []
    vert = []
    for x1,y1,x2,y2 in merged_segments:
        if abs(y1-y2) < 1.0:
            a,b = sorted([x1,x2])
            horiz.append((y1, a, b))
        elif abs(x1-x2) < 1.0:
            a,b = sorted([y1,y2])
            vert.append((x1, a, b))

    # Horizontal pairs (two lines with small delta Y)
    horiz.sort()
    for i in range(len(horiz)):
        y, a1, b1 = horiz[i]
        for j in range(i+1, min(i+15, len(horiz))):
            y2, a2, b2 = horiz[j]
            dy = abs(y2 - y)
            if dy < 1e-6 or dy > delta_pair_px: 
                if dy > delta_pair_px: break
                continue
            # Overlap length
            left = max(a1, a2); right = min(b1, b2)
            if right > left:
                width = right - left
                if min_w_px <= width <= max_w_px:
                    windows.append(("window", left, (y+y2)/2, right, (y+y2)/2))
    # Vertical pairs
    vert.sort()
    for i in range(len(vert)):
        x, a1, b1 = vert[i]
        for j in range(i+1, min(i+15, len(vert))):
            x2, a2, b2 = vert[j]
            dx = abs(x2 - x)
            if dx < 1e-6 or dx > delta_pair_px:
                if dx > delta_pair_px: break
                continue
            top = max(a1, a2); bottom = min(b1, b2)
            if bottom > top:
                height = bottom - top
                if min_w_px <= height <= max_w_px:
                    windows.append(("window", (x+x2)/2, top, (x+x2)/2, bottom))
    return windows

def _rasterize_walls(segments, shape, thickness=5):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for x1,y1,x2,y2 in segments:
        cv2.line(mask, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 255, thickness=thickness)
    return mask

def _find_rooms(mask_walls, min_area_px):
    # Rooms = connected components of free space (invert walls and fill borders)
    h,w = mask_walls.shape
    solid = cv2.copyMakeBorder(mask_walls, 2,2,2,2, cv2.BORDER_CONSTANT, value=255)
    inv = 255 - solid
    num, labels = cv2.connectedComponents(inv)
    rooms = []
    for lab in range(1, num):
        comp = (labels == lab).astype(np.uint8)*255
        # remove border-padding
        comp = comp[2:h+2, 2:w+2]
        area = int(comp.sum()/255)
        if area >= min_area_px:
            cnts,_ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            cnt = max(cnts, key=cv2.contourArea)
            rooms.append(cnt.reshape(-1,2).astype(float).tolist())
    return rooms

# ---------- endpoint ----------

@router.post("/segment/v2")
async def segment_v2(
    file: UploadFile = File(...),
    scale_mm_per_px: float = Query(0.0, description="мм/пикс (0 = пиксельный режим)"),
    debug: bool = Query(True, description="возвращать отладку")
):
    cfg = load_config()

    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img0 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img0 is None:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    img, _ = _resize_max_side(img0, int(cfg["resize"]["max_side_px"]))

    # binarize
    bw = _binarize(img, cfg["binarize"]["method"], int(cfg["binarize"]["block_size"]), int(cfg["binarize"]["C"]))
    if cfg.get("deskew", True):
        bw = _deskew(bw)

    # remove text/dimensions (optional)
    text_mask = None
    if cfg.get("text", {}).get("remove", True):
        bw, text_mask = _remove_text(bw, kernel=int(cfg["text"]["kernel"]), dilate=int(cfg["text"]["dilate"]), inpaint_radius=int(cfg["text"]["inpaint_radius"]))

    # morphology
    k_open = int(cfg["morph"]["open_kernel"]); k_close = int(cfg["morph"]["close_kernel"])
    if k_open > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open)))
    if k_close > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close)))

    # scale
    mm_per_px = float(scale_mm_per_px) if scale_mm_per_px and scale_mm_per_px > 0 else 1.0
    def mm_to_px(mm): return max(1, int(round(mm / mm_per_px)))

    # lines
    seg_raw, edges = _extract_lines(bw, int(cfg["hough"]["threshold"]), mm_to_px(cfg["hough"]["min_line_length_mm"]), mm_to_px(cfg["hough"]["max_line_gap_mm"]))
    snapped = _snap_ortho(seg_raw, angle_tol_deg=float(cfg["hough"]["angle_tolerance_deg"]))
    step_px = mm_to_px(cfg["grid"]["snap_mm"])
    snapped = [(_grid_snap(x1, step_px), _grid_snap(y1, step_px), _grid_snap(x2, step_px), _grid_snap(y2, step_px)) for (x1,y1,x2,y2) in snapped]
    merged = _merge_collinear(snapped, gap_tol_px=mm_to_px(cfg["merge"]["collinear_gap_mm"]), min_seg_px=mm_to_px(cfg["merge"]["min_segment_mm"]))

    # openings
    doors = _detect_openings_doors(merged, min_gap_px=mm_to_px(cfg["doors"]["gap_min_mm"]), max_gap_px=mm_to_px(cfg["doors"]["gap_max_mm"]))
    windows = _detect_windows_from_pairs(merged, min_w_px=mm_to_px(cfg["windows"]["min_width_mm"]), max_w_px=mm_to_px(cfg["windows"]["max_width_mm"]), delta_pair_px=int(cfg["windows"]["pair_delta_px"]))

    # rooms (rasterize walls, then connected components)
    wall_mask = _rasterize_walls(merged, img.shape, thickness=max(3, mm_to_px(50)))  # ~50mm wall thickness on raster
    min_area_px = (mm_to_px(1000) * mm_to_px(1000)) // (mm_to_px(1000))  # fallback approx; will replace below
    # better: convert m2 to px using mm_per_px
    min_area_m2 = float(cfg["rooms"]["min_area_m2"])
    min_area_px = int((min_area_m2 * 1_000_000) / (mm_per_px*mm_per_px))
    rooms_cnts = _find_rooms(wall_mask, min_area_px=min_area_px)

    # Optional OCR (lazy import, may be disabled on server)
    ocr = []
    try:
        import easyocr  # type: ignore
        reader = easyocr.Reader(['ru','en'], gpu=False)
        res = reader.readtext(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for (bbox, text, conf) in res:
            if conf < 0.55: 
                continue
            ocr.append({"text": text, "bbox": bbox, "conf": conf})
    except Exception:
        ocr = []

    # build json
    walls_json = [ dict(x1=float(x1),y1=float(y1),x2=float(x2),y2=float(y2)) for (x1,y1,x2,y2) in merged ]
    openings_json = [ dict(type=t, x1=float(x1),y1=float(y1),x2=float(x2),y2=float(y2)) for (t,x1,y1,x2,y2) in doors + windows ]
    rooms_json = [ dict(polygon=poly) for poly in rooms_cnts ]

    dbg = None
    if debug:
        dbg = dict(
            bw_preview=_img_to_dataurl(bw),
            edges_preview=_img_to_dataurl(edges),
            text_mask=_img_to_dataurl(text_mask) if text_mask is not None else None,
            raw_lines=[ [x1,y1,x2,y2] for (x1,y1,x2,y2) in seg_raw ],
            merged=[ [x1,y1,x2,y2] for (x1,y1,x2,y2) in merged ],
            wall_mask=_img_to_dataurl(wall_mask)
        )

    return JSONResponse({
        "walls": walls_json,
        "openings": openings_json,
        "rooms": rooms_json,
        "ocr": ocr,
        "debug": dbg,
        "scale_used_mm_per_px": mm_per_px
    })
