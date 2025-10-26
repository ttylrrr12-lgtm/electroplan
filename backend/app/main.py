# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from typing import List, Tuple, Literal, Dict
from pydantic import BaseModel, Field

import math
import heapq
import os

try:
    import yaml  # PyYAML
except Exception:
    yaml = None

# 1) Создаём приложение
app = FastAPI(title="ElectroPlan Backend", version="1.1.2")

# 2) CORS (разрешаем фронту ходить к API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # при желании сузить до твоего домена
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3) Подключаем сегментацию v2 (новый эндпоинт /api/segment/v2)
from .segment_v2 import router as segment_v2_router
app.include_router(segment_v2_router)

# (опционально) если хочешь сохранить /api/segment (старый), можно обернуть в try:
try:
    from .segment import router as segment_router  # только если файл есть
    app.include_router(segment_router)
except Exception:
    pass

# ---------- Models ----------
class Wall(BaseModel):
    x1: float; y1: float; x2: float; y2: float

class Opening(BaseModel):
    type: Literal["door","window"]
    x1: float; y1: float; x2: float; y2: float

class Device(BaseModel):
    id: str
    type: Literal["panel","socket","switch","light"]
    x: float; y: float

class RouteOut(BaseModel):
    polyline: List[Tuple[float,float]]
    type: Literal["power","light"]
    toDeviceId: str

class RouteRequestV1(BaseModel):
    scale_mm_per_px: float = Field(50, gt=0)
    walls: List[Wall] = Field(default_factory=list)
    devices: List[Device] = Field(default_factory=list)

class RouteResponse(BaseModel):
    routes: List[RouteOut] = Field(default_factory=list)
    warnings: List[Dict] = Field(default_factory=list)

class RouteRequestV2(BaseModel):
    scale_mm_per_px: float = Field(50, gt=0)
    ceiling_mm: int = 2700
    wall_material: Literal["brick","concrete","wood"] = "brick"
    ruleset: str = "PUE_RU_v1"
    walls: List[Wall] = Field(default_factory=list)
    openings: List[Opening] = Field(default_factory=list)
    devices: List[Device] = Field(default_factory=list)

# ---------- Utils ----------
def segs_intersect(ax,ay,bx,by, cx,cy,dx,dy):
    def ccw(x1,y1,x2,y2,x3,y3): return (y3-y1)*(x2-x1) > (y2-y1)*(x3-x1)
    return (ccw(ax,ay,cx,cy,dx,dy) != ccw(bx,by,cx,cy,dx,dy)) and (ccw(ax,ay,bx,by,cx,cy) != ccw(ax,ay,bx,by,dx,dy))

def dist(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])

# ---------- Health ----------
@app.get("/health")
def health(): 
    return {"status":"ok"}

# ---------- v1 simple L routing (backward compat) ----------
@app.post("/api/route", response_model=RouteResponse)
def api_route(req: RouteRequestV1):
    devices = req.devices; walls = req.walls
    panels = [d for d in devices if d.type=="panel"]
    if not panels:
        return RouteResponse(routes=[], warnings=[{"code":"NO_PANEL","message":"Установите щиток (panel)"}])
    panel = panels[0]

    def count_crossings(poly, walls):
        c=0
        for i in range(len(poly)-1):
            x1,y1 = poly[i]; x2,y2 = poly[i+1]
            for w in walls:
                if segs_intersect(x1,y1,x2,y2, w.x1,w.y1,w.x2,w.y2): 
                    c+=1
        return c

    routes: List[RouteOut] = []; warnings=[]
    for d in devices:
        if d.type=="panel": continue
        elbow1 = (panel.x, d.y); elbow2 = (d.x, panel.y)
        poly1 = [(panel.x,panel.y), elbow1, (d.x,d.y)]
        poly2 = [(panel.x,panel.y), elbow2, (d.x,d.y)]
        c1 = count_crossings(poly1, walls); c2 = count_crossings(poly2, walls)
        poly = poly1 if c1 <= c2 else poly2
        # проверим ортогональность
        for i in range(len(poly)-1):
            x1,y1=poly[i]; x2,y2=poly[i+1]
            if x1!=x2 and y1!=y2: 
                warnings.append({"code":"NON_ORTHO","message":"Найден неортогональный сегмент — проверьте трассу."})
                break
        routes.append(RouteOut(polyline=poly, type=("light" if d.type=="light" else "power"), toDeviceId=d.id))
    return RouteResponse(routes=routes, warnings=warnings)

# ---------- v2 routing: channels + A* ----------
def _default_rules():
    return {
        "channels": {"default_offset_from_wall_mm": 30},
        "forbidden_zones": [{"type": "opening", "margin_mm": 100}],
        "costs": {"per_meter": 1.0}
    }

def load_rules(ruleset_key: str):
    # ожидаем rules/<ruleset>.yaml рядом с этим файлом
    base_dir = os.path.dirname(__file__)
    rules_dir = os.path.join(base_dir, "rules")
    path = os.path.join(rules_dir, f"{ruleset_key}.yaml")
    fallback = os.path.join(rules_dir, "PUE_RU_v1.yaml")
    if yaml is None:
        return _default_rules()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or _default_rules()
        if os.path.exists(fallback):
            with open(fallback, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or _default_rules()
    except Exception:
        pass
    return _default_rules()

def offset_segment(seg, offset_px):
    (x1,y1,x2,y2) = seg
    vx, vy = x2-x1, y2-y1
    L = math.hypot(vx,vy)
    if L == 0: return seg
    nx, ny = -vy/L, vx/L  # ед. нормаль
    return (x1+nx*offset_px, y1+ny*offset_px, x2+nx*offset_px, y2+ny*offset_px)

def build_channel_nodes(walls, mm_per_px, default_offset_mm=30, step_px=30):
    offset_px = max(1.0, default_offset_mm / mm_per_px) if mm_per_px>0 else 10.0
    nodes = []
    edges = []
    for w in walls:
        ox1,oy1,ox2,oy2 = offset_segment((w.x1,w.y1,w.x2,w.y2), offset_px)
        length = math.hypot(ox2-ox1, oy2-oy1)
        steps = max(1, int(length/step_px))
        line_nodes = []
        for i in range(steps+1):
            t = i/steps
            x = ox1 + (ox2-ox1)*t
            y = oy1 + (oy2-oy1)*t
            idx = len(nodes)
            nodes.append((x,y))
            line_nodes.append(idx)
        for i in range(len(line_nodes)-1):
            a = line_nodes[i]; b = line_nodes[i+1]
            edges.append((a,b)); edges.append((b,a))
    return nodes, edges

def add_device_connectors(nodes, edges, devices, step_px=30):
    for d in devices:
        best_i = None; best_dist = 1e18; best_poly=None
        for i, (x,y) in enumerate(nodes):
            poly1 = [(d.x,d.y),(x,d.y),(x,y)]
            poly2 = [(d.x,d.y),(d.x,y),(x,y)]
            L1 = dist(poly1[0], poly1[1]) + dist(poly1[1], poly1[2])
            L2 = dist(poly2[0], poly2[1]) + dist(poly2[1], poly2[2])
            L = min(L1, L2); poly = poly1 if L1<=L2 else poly2
            if L < best_dist: best_dist=L; best_i=i; best_poly=poly
        if best_i is not None and best_poly:
            elbow = best_poly[1]
            idx_elbow = len(nodes); nodes.append((elbow[0], elbow[1]))
            edges.append((idx_elbow, best_i)); edges.append((best_i, idx_elbow))
            idx_dev = len(nodes); nodes.append((d.x, d.y))
            edges.append((idx_dev, idx_elbow)); edges.append((elbow := idx_elbow, idx_dev))
            d._graph_node = idx_dev
        else:
            d._graph_node = None

def forbid_edges_by_openings(nodes, edges, openings, mm_per_px, margin_mm=100):
    margin_px = (margin_mm / mm_per_px) if mm_per_px>0 else 0.0
    def near_segment(x1,y1,x2,y2, sx,sy,tx,ty, tol):
        def point_seg_dist(px,py, ax,ay,bx,by):
            vx,vy = bx-ax, by-ay
            L2 = vx*vx+vy*vy
            if L2==0: return math.hypot(px-ax, py-ay)
            t=max(0,min(1, ((px-ax)*vx+(py-ay)*vy)/L2))
            projx = ax + t*vx; projy = ay + t*vy
            return math.hypot(px-projx, py-projy)
        d1 = point_seg_dist(x1,y1, sx,sy,tx,ty)
        d2 = point_seg_dist(x2,y2, sx,sy,tx,ty)
        return (d1<tol) or (d2<tol)
    keep = []
    for (a,b) in edges:
        x1,y1 = nodes[a]; x2,y2 = nodes[b]
        blocked = False
        for o in openings:
            if near_segment(x1,y1,x2,y2, o.x1,o.y1,o.x2,o.y2, margin_px):
                blocked = True; break
        if not blocked: keep.append((a,b))
    return keep

def astar(nodes, edges, start_idx, goal_idx, per_meter_cost=1.0):
    adj = {}
    for a,b in edges:
        adj.setdefault(a, []).append(b)
    def heuristic(i,j): return dist(nodes[i], nodes[j])
    INF = 1e18
    g = {start_idx: 0.0}
    f = {start_idx: heuristic(start_idx, goal_idx)}
    parent = {}
    pq = [(f[start_idx], start_idx)]
    visited = set()
    while pq:
        _, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u)
        if u == goal_idx: break
        for v in adj.get(u, []):
            w = dist(nodes[u], nodes[v]) * per_meter_cost
            tentative = g[u] + w
            if tentative < g.get(v, INF):
                g[v] = tentative
                parent[v] = u
                f[v] = tentative + heuristic(v, goal_idx)
                heapq.heappush(pq, (f[v], v))
    if goal_idx not in parent and goal_idx != start_idx:
        return None
    path = [goal_idx]
    cur = goal_idx
    while cur != start_idx:
        cur = parent.get(cur)
        if cur is None: return None
        path.append(cur)
    path.reverse()
    return [ (nodes[i][0], nodes[i][1]) for i in path ]

@app.post("/api/route/v2", response_model=RouteResponse)
def api_route_v2(req: RouteRequestV2):
    rules = load_rules(req.ruleset)
    mm_per_px = req.scale_mm_per_px or 1.0

    # 1) Каналы вдоль стен
    nodes, edges = build_channel_nodes(
        req.walls, 
        mm_per_px, 
        default_offset_mm=rules["channels"].get("default_offset_from_wall_mm", 30), 
        step_px=30
    )

    # 2) Запретные зоны возле проёмов
    margin_mm = 100
    if rules.get("forbidden_zones") and isinstance(rules["forbidden_zones"], list):
        margin_mm = rules["forbidden_zones"][0].get("margin_mm", 100)
    edges = forbid_edges_by_openings(nodes, edges, req.openings, mm_per_px, margin_mm=margin_mm)

    # 3) Коннекторы устройств к ближайшим каналам
    add_device_connectors(nodes, edges, req.devices, step_px=30)

    # 4) Точка старта — щиток
    panels = [d for d in req.devices if d.type=="panel"]
    if not panels:
        return RouteResponse(routes=[], warnings=[{"code":"NO_PANEL","message":"Установите щиток (panel)"}])
    panel = panels[0]
    if getattr(panel, "_graph_node", None) is None:
        return RouteResponse(routes=[], warnings=[{"code":"NO_PANEL_CONNECT","message":"Щиток не удалось привязать к каналу — добавьте стены ближе."}])

    # 5) Прокладка A*
    routes = []; warnings=[]
    per_meter = rules.get("costs",{}).get("per_meter", 1.0)
    # приводим стоимость к "за пиксель"
    per_pixel_cost = per_meter / (1000.0 / mm_per_px)

    for d in req.devices:
        if d.type=="panel": continue
        if getattr(d, "_graph_node", None) is None:
            warnings.append({"code":"NO_DEVICE_CONNECT","message":f"Устройство {d.id} не привязано к каналу — слишком далеко."})
            continue
        path = astar(nodes, edges, start_idx=panel._graph_node, goal_idx=d._graph_node, per_meter_cost=per_pixel_cost)
        if path is None:
            warnings.append({"code":"NO_PATH","message":f"Не найден путь до устройства {d.id}"})
            continue
        routes.append(RouteOut(polyline=path, type=("light" if d.type=="light" else "power"), toDeviceId=d.id))

    return RouteResponse(routes=routes, warnings=warnings)
