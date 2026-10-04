FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# build-essential 用于少数依赖的本地构建，libgomp1 是 CPU 推理运行时需要的 OpenMP 库；
# 安装完成后立即清理 apt 索引，避免把构建缓存带入生产镜像。
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.4.1+cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app

# UID 1000 与宿主机常见开发用户保持一致，避免服务进程以 root 身份运行。
RUN groupadd --gid 1000 shopcare \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin shopcare \
    && mkdir -p /app/.data \
    && chown -R shopcare:shopcare /app

USER shopcare

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
