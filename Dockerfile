FROM python:3.11-slim

# 1) 시스템 의존성 설치  ── Open3D(OpenMP), OpenCV 등
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2) 작업 디렉터리
WORKDIR /app

# 3) 파이썬 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) 애플리케이션 복사
COPY . .

# 5) MCP 서버 실행
CMD ["python", "mcp_server.py"]