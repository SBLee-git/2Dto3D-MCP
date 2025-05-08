FROM python:3.11-slim

# 1) 시스템 의존성 설치 (OpenCV, Open3D 실행에 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# 2) 작업 디렉토리 설정
WORKDIR /app

# 3) requirements.txt 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) 애플리케이션 소스 복사
COPY . .

# 5) static 폴더 미리 생성
RUN mkdir -p static

# 6) 포트 노출
EXPOSE 8001

# 7) Smithery용 환경변수 (기본값 제공)
ENV STATIC_URL_BASE=/static

# 8) FastAPI 서버 실행 (Smithery에서 이 포트 매핑함)
CMD ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8001"]
