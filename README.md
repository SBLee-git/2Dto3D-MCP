# 2Dto3D-MCP

2D 이미지를 3D 메시(mesh)로 변환하는 서버 애플리케이션  
REST API를 통해 2D 데이터를 입력받아 3D 모델 파일(STL, OBJ 등)로 변환해주는 서비스입니다.

---

## 🧑‍💻 내 역할 (My Role)
- **백엔드 개발**: FastAPI 기반 서버 구현 및 2D→3D 변환 로직 개발
- **패키지 관리 및 환경 구성**: `uv`, `pyproject.toml` 등 최신 Python 패키지 관리 도구 적용
- **배포 자동화**: Docker 및 Smithery를 활용한 CI/CD 파이프라인 구축

---

## 🛠️ 기술 스택 (Tech Stack)
- **언어**: Python 3.11
- **웹 프레임워크**: FastAPI
- **3D 처리 라이브러리**: Open3D
- **패키지/환경 관리**: uv, pyproject.toml, requirements.txt
- **컨테이너**: Docker
- **CI/CD**: Smithery
- **기타**: uv.lock, runtime.txt 등

---
## 🚀 서비스 개요

- **2D 이미지를 업로드하면 3D 메시 파일로 변환하여 반환**
- OBJ 3D 포맷 지원
- REST API 기반으로 외부 서비스와 연동 가능
- Docker 컨테이너로 손쉽게 배포 및 확장 가능

---
## 📡 API 예시

> 실제 엔드포인트 및 파라미터는 mcp_server.py를 참고하세요.

- **2D 이미지 업로드 및 3D 변환**
    ```
    POST /convert
    Content-Type: multipart/form-data

    - file: 2D 이미지 파일 (png, jpg 등)
    - output_format: obj
    ```

- **응답**
    - 변환된 3D 파일 다운로드

---
## 💡 기타

- `pyproject.toml`과 `uv.lock`을 통해 최신 Python 생태계에 맞춘 패키지 관리
- Docker로 손쉬운 배포 및 운영 자동화
- FastAPI 기반으로 확장성과 유지보수성이 뛰어난 구조
