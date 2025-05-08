import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi_mcp import FastApiMCP

# ✅ 정적 파일 디렉토리 설정
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# ✅ FastAPI + MCP 초기화
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ✅ MCP 객체 생성 및 mount
mcp = FastApiMCP(app)
mcp.mount()

# ✅ MCP 툴 등록
from mcp_tool import register_tools
register_tools(mcp)

# ✅ Smithery 인식용 FastAPI 객체 내보내기
__all__ = ["app"]
