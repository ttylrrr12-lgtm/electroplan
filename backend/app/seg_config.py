
import os, yaml

_DEFAULT = {
    "deskew": True,
    "resize": {"max_side_px": 2200},
    "binarize": {"method": "adaptive_gaussian", "block_size": 31, "C": 5},
    "morph": {"open_kernel": 3, "close_kernel": 7},
    "hough": {"min_line_length_mm": 250, "max_line_gap_mm": 60, "angle_tolerance_deg": 10, "threshold": 75},
    "grid": {"snap_mm": 5},
    "merge": {"collinear_gap_mm": 70, "min_segment_mm": 180},
    "doors": {"gap_min_mm": 700, "gap_max_mm": 1200},
    "windows": {"min_width_mm": 600, "max_width_mm": 2400, "pair_delta_px": 6},
    "text": {"remove": true, "kernel": 3, "dilate": 2, "inpaint_radius": 2},
    "rooms": {"min_area_m2": 1.5, "polygon_simplify_epsilon_mm": 50}
}

def load_config() -> dict:
    candidate = os.path.join(os.getcwd(), "rules", "segmentation.yaml")
    cfg = {}
    if os.path.exists(candidate):
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
    def deep(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    return deep(dict(_DEFAULT), cfg)
