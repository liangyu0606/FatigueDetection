# 使用Python 3.10官方镜像作为基础镜像，支持ARM64架构
FROM --platform=linux/arm64 python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    # 基础构建工具
    build-essential \
    cmake \
    pkg-config \
    # OpenCV依赖
    libopencv-dev \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    # dlib依赖
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    # 音频支持
    libasound2-dev \
    libpulse-dev \
    libsdl2-dev \
    libsdl2-mixer-2.0-0 \
    # FFmpeg支持
    ffmpeg \
    # 网络和SSL
    libssl-dev \
    libffi-dev \
    # 清理缓存
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 升级pip并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# 创建必要的目录结构
RUN mkdir -p /app/output/models \
    /app/output/logs \
    /app/output/processed_data \
    /app/static \
    /app/templates

# 复制项目文件
COPY . .

# 设置权限
RUN chmod +x fatigue_web_ffmpeg.py

# 创建非root用户
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "fatigue_web_ffmpeg:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
