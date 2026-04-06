import tensorrt as trt
import os

def build_engine(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.builder_optimization_level = 1

    # --- 核心修复：针对 RTX 5060 Ti 的强力限制 ---
    # 强制只使用传统的 GPU 算子源，绝对禁用 Myelin (ForeignNode)
    # TensorRT 10 中 Myelin 往往包含在默认策略里，我们必须显式排除
    config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS) | 
                              1 << int(trt.TacticSource.CUBLAS_LT) | 
                              1 << int(trt.TacticSource.CUDNN))
    
    # 如果环境允许，强制禁用预览特性
    try:
        config.set_preview_configuration(trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805, False)
    except:
        pass

    # 配置优化配置文件
    profile = builder.create_optimization_profile()
    input_name = "input" 
    input_shape = (1, 3, 224, 224)
    
    print(f"📦 正在解析 ONNX 文件: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return

    # 动态匹配输入名
    real_input_name = network.get_input(0).name
    profile.set_shape(real_input_name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)

    # 开启 FP16 
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    print("🛠️  正在针对 RTX 5060 Ti 进行强制算法对齐构建...")
    plan = builder.build_serialized_network(network, config)
    
    if plan:
        with open(engine_path, 'wb') as f:
            f.write(plan)
        print(f"✅ 成功生成 Engine: {engine_path}")
    else:
        print("❌ 构建失败：TensorRT 仍无法在 SM 10.0 上找到匹配算法。")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    build_engine(os.path.join(current_dir, "model.onnx"), os.path.join(current_dir, "model.engine"))