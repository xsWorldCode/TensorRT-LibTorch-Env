import onnxruntime as ort
import numpy as np
import time

# 1. 设置推理配置
# 显式指定 CUDA 12.8 驱动及参数
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'DEFAULT',
        'do_copy_in_default_stream': True,
    })
]

try:
    # 加载模型
    session = ort.InferenceSession("model.onnx", providers=providers)
    print("✅ RTX 5060 Ti (SM 120) 加速引擎已就绪！")
except Exception as e:
    print(f"❌ 初始化失败: {e}")
    exit()

# 2. 构造固定尺寸的随机输入
# 这里手动指定 [Batch_Size, Channel, Height, Width]
# 如果是图像分类通常是 224，如果是目标检测(YOLO)通常是 640
input_name = session.get_inputs()[0].name
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 3. 预热 (Warm-up)
# 50 系列显卡在初次推理时会进行算子调优，必须预热
print("正在执行 20 次推理预热...")
for _ in range(20):
    session.run(None, {input_name: dummy_input})

# 4. 性能压力测试
print("开始 500 次压力推理测试...")
start_time = time.perf_counter()
for _ in range(500):
    session.run(None, {input_name: dummy_input})
end_time = time.perf_counter()

# 5. 计算结果
total_time = end_time - start_time
avg_latency = (total_time / 500) * 1000
fps = 500 / total_time

print("\n" + "="*40)
print(f"🚀 RTX 5060 Ti 性能评估报告")
print(f"硬件架构: Blackwell (SM 120)")
print(f"平均推理耗时: {avg_latency:.3f} ms")
print(f"实时帧率 (FPS): {fps:.2f}")
print("="*40)
