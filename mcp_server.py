import os
import sys
import base64
import hashlib  # 캐싱용 해시
import zipfile
import shutil
import numpy as np
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel

# ────────────────────────────────
# 1. FastAPI & MCP 서버 초기화
# ────────────────────────────────
app = FastAPI(
    title="2Dto3D MCP Server",
    version="1.0.3",
    description="Convert 2D map images (base64) to zipped 3D OBJ files using MCP."
)
mcp = FastApiMCP(app)  # MCP 서버 인스턴스 생성

# ────────────────────────────────
# 2. 입력 스키마(pydantic)
# ────────────────────────────────
class ImagePayload(BaseModel):
    base64_image: str

# ────────────────────────────────
# 3. MCP Tool로 변환 함수 등록
#    (Smithery가 자동 인식하는 함수)
# ────────────────────────────────
@mcp.tool()  # 이게 반드시 필요!
def convert_map(payload: ImagePayload) -> str:
    """
    2D 이미지를 3D 벽체 OBJ 파일(zip)로 변환하고, URL을 반환하는 MCP Tool 함수
    입력: payload (base64_image)
    출력: static 디렉토리에 저장된 zip 파일의 URL(string)
    """
    try:
        import cv2
        import open3d as o3d
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        STATIC_DIR = "static"
        os.makedirs(STATIC_DIR, exist_ok=True)

        # ── 이미지 디코드 및 캐싱 ──
        img_bytes = base64.b64decode(payload.base64_image)
        hash_key = hashlib.md5(img_bytes).hexdigest()
        file_id = f"map_walls_only_{hash_key}.zip"
        zip_static_path = os.path.join(STATIC_DIR, file_id)
        static_url_base = os.getenv("STATIC_URL_BASE", "/static")
        cached_url = f"{static_url_base.rstrip('/')}/{file_id}"
        if os.path.exists(zip_static_path):
            print(f"[CACHE] Returning cached result: {file_id}", file=sys.stderr)
            return cached_url

        # ── 이미지 읽기 및 전처리 ──
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image.")

        cm_per_pixel = 1.0
        wall_height = 200.0
        wall_thick = 2

        temp_dir = "mcp_temp"
        os.makedirs(temp_dir, exist_ok=True)
        zip_path_temp = os.path.join(temp_dir, file_id)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        ker1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wall_thick, wall_thick))
        ker2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(edges, ker1, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker2, iterations=2)

        n_labels, labels = cv2.connectedComponents(mask)
        parts = []

        # ── 각 라벨마다 contour 추출 및 폴리곤 병합 ──
        for lid in range(1, n_labels):
            comp_mask = (labels == lid).astype(np.uint8) * 255
            if cv2.countNonZero(comp_mask) < 20:
                continue
            contours, hierarchy = cv2.findContours(comp_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None:
                continue
            polys = []
            for i, cnt in enumerate(contours):
                ext = [tuple(p[0]) for p in cnt]
                holes = []
                child = hierarchy[0][i][2]
                while child != -1:
                    holes.append([tuple(p[0]) for p in contours[child]])
                    child = hierarchy[0][child][0]
                polys.append(Polygon(ext, holes))
            from shapely.ops import unary_union
            merged = unary_union(polys)
            verts_local, faces_local = [], []
            offset_local = 0
            exterior = list(merged.exterior.coords)
            for j in range(len(exterior)):
                x0,y0 = exterior[j]; x1,y1 = exterior[(j+1)%len(exterior)]
                verts_local.extend([[x0,y0,0],[x1,y1,0],[x1,y1,wall_height],[x0,y0,wall_height]])
                faces_local.extend([
                    [offset_local, offset_local+1, offset_local+2],
                    [offset_local+2, offset_local+3, offset_local]
                ])
                offset_local += 4
            # 내부 홀 처리
            for interior in merged.interiors:
                hole_coords = list(interior.coords)
                for k in range(len(hole_coords)):
                    x0,y0 = hole_coords[k]; x1,y1 = hole_coords[(k+1)%len(hole_coords)]
                    verts_local.extend([[x0,y0,0],[x1,y1,0],[x1,y1,wall_height],[x0,y0,wall_height]])
                    faces_local.extend([
                        [offset_local, offset_local+2, offset_local+1],
                        [offset_local+2, offset_local+3, offset_local]
                    ])
                    offset_local += 4
            if verts_local:
                parts.append((verts_local, faces_local))

        # ── 벡터가 없으면 빈 zip 반환 ──
        if not parts:
            with zipfile.ZipFile(zip_path_temp, "w") as zf:
                zf.writestr("empty.txt", "No geometry was generated.")
            shutil.move(zip_path_temp, zip_static_path)
            return cached_url

        # ── 개별 OBJ 생성 및 zip 압축 ──
        def mesh_to_obj_string(mesh):
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            lines = [f"v {x} {y} {z}" for x, y, z in verts]
            lines += [f"f {a+1} {b+1} {c+1}" for a, b, c in faces]
            return "\n".join(lines)

        with zipfile.ZipFile(zip_path_temp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for idx, (verts, faces) in enumerate(parts):
                mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(np.array(verts) * cm_per_pixel),
                    triangles=o3d.utility.Vector3iVector(np.array(faces, dtype=np.int32))
                )
                mesh.compute_vertex_normals()
                obj_str = mesh_to_obj_string(mesh)
                zf.writestr(f"wall_{idx}.obj", obj_str)

        shutil.move(zip_path_temp, zip_static_path)
        print(f"[API] Result URL: {cached_url}", file=sys.stderr)
        return cached_url

    except ImportError as ie:
        error_message = f"Import error: {str(ie)}."
        print(f"[MCP ERROR] {error_message}", file=sys.stderr)
        return f"[ERROR] {error_message}"
    except Exception as e:
        import traceback
        error_message = f"Error: {type(e).__name__}: {e}"
        print(f"[MCP ERROR] {error_message}\n{traceback.format_exc()}", file=sys.stderr)
        return f"[ERROR] {error_message}"

# ────────────────────────────────
# 4. MCP 엔드포인트 등록 (필수!)
# ────────────────────────────────
mcp.mount()  # 이걸 꼭 마지막에! (Tool Discovery 가능)

# ────────────────────────────────
# 5. HealthCheck 엔드포인트 (컨테이너용)
# ────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ────────────────────────────────
# 6. 직접 실행용 (개발·로컬테스트)
# ────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
