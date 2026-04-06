# 使用 NVIDIA 官方 TensorRT 鏡像
FROM nvcr.io/nvidia/tensorrt:24.01-py3

WORKDIR /app

# 安裝 OpenCV 運行庫
RUN apt-get update && apt-get install -y libopencv-dev

# 拷貝編譯產物與模型 (注意路徑需匹配實體位置)
COPY ./build/env_check /app/
COPY ./models /app/models/
COPY ./build/test.jpg /app/test.jpg

# 確保環境變數包含 ONNX Runtime 的庫路徑
# 注意：在容器內，路徑需指向容器內部的位置
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 啟動程式
CMD ["./env_check"]
