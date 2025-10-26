
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Literal, Dict, Tuple
import numpy as np
import cv2

router = APIRouter(prefix="/api", tags=["segmentation"])

class WallOut(BaseModel):
    x1: float; y1: float; x2: float; y2: float

class OpeningOut(BaseModel):
    type: Literal["door","window"]
    x1: float; y1: float; x2: float; y2: float

def _read_image_bytes(file: UploadFile) -> np.ndarray:
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Unsupported image format")
    return img

def _binarize(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 5)
    return bw

def _deskew(bw: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(bw, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is None:
        return bw
    angles = []
    for rho, theta in lines[:,0,:]:
        a = (theta * 180 / np.pi) % 180
        if a > 135: a -= 180
        angles.append(a)
    if not angles:
        return bw
    median = float(np.median(angles))
    target = min([ -90, -45, 0, 45, 90 ], key=lambda t: abs(median - t))
    delta = median - target
    if abs(delta) < 0.5:
        return bw
    h, w = bw.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), delta, 1.0)
    rot = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return rot

def _extract_lines(bw: np.ndarray):
    edges = cv2.Canny(bw, 60, 160, apertureSize=3)
    h, w = bw.shape
    min_len = max(30, int(min(h, w)*0.03))
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=min_len, maxLineGap=8)
    segments = []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            segments.append((float(x1),float(y1),float(x2),float(y2)))
    return segments

def _snap_ortho(segments, angle_tol_deg=10):
    snapped = []
    for x1,y1,x2,y2 in segments:
        dx, dy = x2-x1, y2-y1
        ang = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0
        if min(abs(ang-0), abs(ang-180)) < angle_tol_deg:
            ym = (y1 + y2)/2.0
            snapped.append((x1, ym, x2, ym))
        elif min(abs(ang-90), abs(ang-270)) < angle_tol_deg or abs(ang-90) < angle_tol_deg:
            xm = (x1 + x2)/2.0
            snapped.append((xm, y1, xm, y2))
        else:
            snapped.append((x1,y1,x2,y2))
    return snapped

def _merge_collinear(segments, gap_tol=15):
    horiz = {}
    vert = {}
    for x1,y1,x2,y2 in segments:
        if abs(y1-y2) < 1.0:
            y = round((y1+y2)/2)
            a,b = sorted([x1,x2])
            horiz.setdefault(y, []).append([a,b])
        elif abs(x1-x2) < 1.0:
            x = round((x1+x2)/2)
            a,b = sorted([y1,y2])
            vert.setdefault(x, []).append([a,b])
    merged = []
    for y, spans in horiz.items():
        spans.sort()
        cur = list(spans[0])
        for a,b in spans[1:]:
            if a - cur[1] <= gap_tol:
                cur[1] = max(cur[1], b)
            else:
                merged.append((cur[0], y, cur[1], y)); cur = [a,b]
        merged.append((cur[0], y, cur[1], y))
    for x, spans in vert.items():
        spans.sort()
        cur = list(spans[0])
        for a,b in spans[1:]:
            if a - cur[1] <= gap_tol:
                cur[1] = max(cur[1], b)
            else:
                merged.append((x, cur[0], x, cur[1])); cur = [a,b]
        merged.append((x, cur[0], x, cur[1]))
    out = []
    for x1,y1,x2,y2 in merged:
        if ((x2-x1)**2 + (y2-y1)**2) >= 24*24:
            out.append((float(x1),float(y1),float(x2),float(y2)))
    return out

def _detect_doors_from_gaps(merged_segments, min_gap=40, max_gap=200):
    doors = []
    horiz_by_y = {}
    vert_by_x = {}
    for x1,y1,x2,y2 in merged_segments:
        if abs(y1-y2) < 1.0:
            y = y1; a,b = sorted([x1,x2])
            horiz_by_y.setdefault(y, []).append((a,b))
        elif abs(x1-x2) < 1.0:
            x = x1; a,b = sorted([y1,y2])
            vert_by_x.setdefault(x, []).append((a,b))
    for y, spans in horiz_by_y.items():
        spans.sort()
        for i in range(len(spans)-1):
            a1,b1 = spans[i]; a2,b2 = spans[i+1]
            gap = a2 - b1
            if min_gap <= gap <= max_gap:
                doors.append((b1, y, a2, y))
    for x, spans in vert_by_x.items():
        spans.sort()
        for i in range(len(spans)-1):
            a1,b1 = spans[i]; a2,b2 = spans[i+1]
            gap = a2 - b1
            if min_gap <= gap <= max_gap:
                doors.append((x, b1, x, a2))
    openings = [("door", float(x1),float(y1),float(x2),float(y2)) for (x1,y1,x2,y2) in doors]
    return openings

@router.post("/segment")
async def segment(file: UploadFile = File(...)):
    img = _read_image_bytes(file)
    bw = _binarize(img)
    bw = _deskew(bw)
    segments = _extract_lines(bw)
    snapped = _snap_ortho(segments, angle_tol_deg=10)
    merged = _merge_collinear(snapped, gap_tol=15)
    openings_raw = _detect_doors_from_gaps(merged, min_gap=40, max_gap=200)

    walls_json = [ dict(x1=x1,y1=y1,x2=x2,y2=y2) for (x1,y1,x2,y2) in merged ]
    openings_json = [ dict(type=t, x1=x1,y1=y1,x2=x2,y2=y2) for (t,x1,y1,x2,y2) in openings_raw ]

    return {
        "walls": walls_json,
        "openings": openings_json,
        "debug": {
            "count_raw": len(segments),
            "count_walls": len(walls_json),
            "count_openings": len(openings_json)
        }
    }
