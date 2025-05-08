import os
import cv2
import base64
import zipfile
import shutil
import uuid
import numpy as np
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi_mcp import FastApiMCP
from shapely.geometry import Polygon
import mapbox_earcut as earcut
import open3d as o3d
from pydantic import BaseModel

# ✅ FastAPI 인스턴스 생성
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ✅ MCP 서버 마운트
mcp = FastApiMCP(app)
mcp.mount()

# ✅ 이미지 디코딩 함수
def decode_base64_image(base64_str: str) -> np.ndarray:
    try:
        img_bytes = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] 이미지 디코딩 실패: {str(e)}", file=sys.stderr)
        raise

# ✅ Open3D 메시 → OBJ 문자열 변환
def mesh_to_obj_string(mesh: o3d.geometry.TriangleMesh) -> str:
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    lines = [f"v {x} {y} {z}" for x, y, z in verts]
    lines += [f"f {a+1} {b+1} {c+1}" for a, b, c in faces]
    return "\n".join(lines)

# ✅ zip 생성 및 반환 URL 처리 (환경변수 사용)
def generate_zip_url(img: np.ndarray) -> str:
    cm_per_pixel = 1.0
    wall_height = 200
    wall_thick = 2

    temp_dir = "mcp_temp"
    os.makedirs(temp_dir, exist_ok=True)

    file_id = f"map_{uuid.uuid4().hex}.zip"
    zip_path = os.path.join(temp_dir, file_id)
    zip_static_path = os.path.join(STATIC_DIR, file_id)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    ker1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wall_thick, wall_thick))
    ker2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(edges, ker1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker2, iterations=2)

    n_labels, labels = cv2.connectedComponents(mask)
    polygons = []

    for lid in range(1, n_labels):
        comp = (labels == lid).astype(np.uint8) * 255
        if cv2.countNonZero(comp) < 20:
            continue
        cnts, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        for idx, cnt in enumerate(cnts):
            if hier[0][idx][3] != -1:
                continue
            ext = cnt.reshape(-1, 2)
            holes = []
            child = hier[0][idx][2]
            while child != -1:
                holes.append(cnts[child].reshape(-1, 2))
                child = hier[0][child][0]
            poly = Polygon(ext, holes)
            if poly.is_valid and poly.area > 1.0:
                polygons.append(poly)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, poly in enumerate(polygons):
            ext = list(poly.exterior.coords)[:-1]
            holes = [list(h.coords)[:-1] for h in poly.interiors]
            all_pts = ext.copy()
            ring_ends = [len(ext)]
            for hole in holes:
                all_pts.extend(hole)
                ring_ends.append(len(all_pts))
            pts2d = np.array(all_pts, dtype=np.float32)

            tri_idx = earcut.triangulate_float32(pts2d, ring_ends)
            verts = [[x, y, 0.0] for x, y in pts2d]
            faces = [[tri_idx[k], tri_idx[k+1], tri_idx[k+2]] for k in range(0, len(tri_idx), 3)]

            for j in range(len(ext)):
                x0, y0 = ext[j]
                x1, y1 = ext[(j+1) % len(ext)]
                base = len(verts)
                verts.extend([
                    [x0, y0, 0.0], [x1, y1, 0.0],
                    [x1, y1, wall_height], [x0, y0, wall_height]
                ])
                faces.extend([[base, base+1, base+2], [base+2, base+3, base+0]])

            mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(np.array(verts) * cm_per_pixel),
                triangles=o3d.utility.Vector3iVector(np.array(faces, dtype=np.int32))
            )
            mesh.compute_vertex_normals()
            obj_str = mesh_to_obj_string(mesh)
            obj_fname = f"wall_{i}.obj"
            obj_path = os.path.join(temp_dir, obj_fname)
            with open(obj_path, "w") as f:
                f.write(obj_str)
            zf.write(obj_path, arcname=obj_fname)
            os.remove(obj_path)

    shutil.move(zip_path, zip_static_path)

    # ✅ 환경변수 기반 반환 URL 처리
    static_url_base = os.getenv("STATIC_URL_BASE", "/static")
    return f"{static_url_base}/{file_id}"

# ✅ API Input 정의
class ImagePayload(BaseModel):
    base64_image: str

@app.post("/convert_map", operation_id="convert_map")
async def convert_map(payload: ImagePayload) -> str:
    print("[MCP] convert_map() 실행", file=sys.stderr)
    try:
        img = decode_base64_image(payload.base64_image)
        return generate_zip_url(img)
    except Exception as e:
        print(f"[MCP ERROR] {str(e)}", file=sys.stderr)
        return f"[ERROR] {str(e)}"

# ✅ 실행 코드
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
