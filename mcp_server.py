import os
import cv2
import base64
import zipfile
import numpy as np
from shapely.geometry import Polygon
import mapbox_earcut as earcut
import open3d as o3d
from mcp.server.fastmcp import FastMCP

# ✅ MCP 서버 초기화
mcp = FastMCP(
    name="map3d",
    instructions="Convert base64 image to multiple 3D OBJ files, zip them, and return as base64 string.",
    host="0.0.0.0",
    port=5173,
    transport="sse"
)

# ✅ Base64 → 이미지 디코딩
def decode_base64_image(base64_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# ✅ 메시 → OBJ 문자열
def mesh_to_obj_string(mesh: o3d.geometry.TriangleMesh) -> str:
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    lines = [f"v {x} {y} {z}" for x, y, z in verts]
    lines += [f"f {a+1} {b+1} {c+1}" for a, b, c in faces]
    return "\n".join(lines)

# ✅ 이미지 → 여러 obj → zip → base64
def generate_zip_base64(img: np.ndarray) -> str:
    cm_per_pixel = 1.0
    wall_height = 200
    wall_thick = 2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wall_thick, wall_thick))
    mask = cv2.dilate(edges, ker, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (3, 3), iterations=2)

    n_labels, labels = cv2.connectedComponents(mask)
    polygons = []

    for lid in range(1, n_labels):
        comp = (labels == lid).astype(np.uint8) * 255
        if cv2.countNonZero(comp) < 20:
            continue
        cnts, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None: continue
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

    temp_dir = "mcp_temp"
    os.makedirs(temp_dir, exist_ok=True)
    zip_path = os.path.join(temp_dir, "output.zip")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, poly in enumerate(polygons):
            ext = list(poly.exterior.coords)
            holes = [list(h.coords) for h in poly.interiors]
            pts2d = np.array(ext + [p for hole in holes for p in hole], np.float32)

            ring_ends, off = [len(ext)], len(ext)
            for hole in holes:
                off += len(hole)
                ring_ends.append(off)

            tri_idx = earcut.triangulate_float32(pts2d, ring_ends)
            verts, faces = [], []
            for x, y in pts2d:
                verts.append([x, y, 0.0])
            for k in range(0, len(tri_idx), 3):
                faces.append([tri_idx[k], tri_idx[k+1], tri_idx[k+2]])
            for ring in [ext] + holes:
                for j in range(len(ring)):
                    x0, y0 = ring[j]
                    x1, y1 = ring[(j+1) % len(ring)]
                    base = len(verts)
                    verts.extend([[x0, y0, 0], [x1, y1, 0], [x1, y1, wall_height], [x0, y0, wall_height]])
                    faces.append([base, base+1, base+2])
                    faces.append([base+2, base+3, base])

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

    # ✅ zip → base64 인코딩
    with open(zip_path, "rb") as zf:
        encoded = base64.b64encode(zf.read()).decode()

    os.remove(zip_path)
    return encoded

# ✅ MCP 도구 등록
@mcp.tool()
async def convert_map(base64_image: str) -> str:
    try:
        img = decode_base64_image(base64_image)
        return generate_zip_base64(img)
    except Exception as e:
        return f"[ERROR] {str(e)}"

# ✅ 실행
if __name__ == "__main__":
    mcp.run(transport="sse")
