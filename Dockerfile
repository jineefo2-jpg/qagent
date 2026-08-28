# 单人部署镜像（2026-08-28）。Python 3.9 对齐本地实测环境 —— ashare/** 全量用
# `from __future__ import annotations` 迁就 3.9 的类型语法，换版本要重跑全套。
FROM python:3.9-slim

# ★ 时区必须显式钉住：ashare 有 6 处 date.today() 取【系统本地时间】判交易日，
#   容器默认 UTC 会让北京时间 00:00-08:00 之间整体错算一天（nightly 判错调仓日）。
ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# duckdb / chromadb / pymupdf 的轮子在 slim 上要编译工具；cron 供每日更新用
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl cron tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# torch 装 CPU 版：默认源会拉 CUDA 依赖，镜像多 3-4 GB 而这里根本没有 GPU
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

COPY . .
# 数据与产出走 volume（见 compose）：镜像不带 4 GB 的 duckdb
VOLUME ["/app/data", "/app/out", "/app/logs", "/app/rag_db"]
EXPOSE 5000
CMD ["python", "server.py"]
