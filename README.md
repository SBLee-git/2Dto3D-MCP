# 🗺️ 2Dto3D-MCP

> **2D 이미지를 업로드하면 3D 메시(OBJ)로 변환해주는 서버 애플리케이션**  
> FastAPI · Open3D · Docker · MCP Protocol 지원

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-%F0%9F%9A%80-brightgreen)
![Docker](https://img.shields.io/badge/docker-ready-2496ed)

---

## ⚡️ TL;DR

- **What**: 2D → 3D 변환 REST API 서버
- **Why**: 게임 맵/시제품을 3D로 빠르게 프로토타이핑
- **Tech**: Python 3.11 · FastAPI · Open3D · Docker

---

## 🧑‍💻 내 역할 (My Role)
- **백엔드 개발**: FastAPI 기반 서버 구현 및 2D→3D 변환 로직 개발
- **패키지 관리 및 환경 구성**: `uv`, `pyproject.toml` 등 최신 Python 패키지 관리 도구 적용
- **배포 자동화**: Docker를 활용한 CI/CD 파이프라인 구축

---

## 🛠️ 기술 스택 (Tech Stack)
- **언어**: Python 3.11
- **웹 프레임워크**: FastAPI
- **3D 처리 라이브러리**: Open3D, NumPy, Shapely, mapbox-earcut
- **패키지/환경 관리**: uv, pyproject.toml, requirements.txt
- **CI/CD**: Docker (+ Smithery 지원)
- **기타**: uv.lock 등

---
## 🚀 서비스 개요

- **2D 이미지를 업로드하면 3D 메시 파일(OBJ)로 변환하여 반환**
- REST API 기반으로 외부 서비스와 연동 가능
- Docker 컨테이너로 손쉽게 배포 및 확장 가능
- OBJ 포맷 지원 (Unreal/Unity 등에서 바로 import)

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
