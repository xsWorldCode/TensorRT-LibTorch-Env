#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include "InferenceEngine.hpp"

int main() {
    // 强制使用绝对路径
    std::string model_path = "/root/libtorch_env/models/model.pt"; 
    
    std::ifstream f(model_path.c_str());
    if (!f.good()) {
        std::cerr << "❌ 错误：在路径 " << model_path << " 找不到模型文件！" << std::endl;
        return -1;
    }

    // 基础环境确认
    std::cout << "GPU 状态确认: " << (torch::cuda::is_available() ? "CUDA 可用" : "CUDA 不可用") << std::endl;
    if (torch::cuda::is_available()) {
        std::cout << "当前 GPU 数量: " << torch::cuda::device_count() << std::endl;
    }

    ResNetPredictor predictor(model_path);

    // 构造模拟图像
    cv::Mat frame(224, 224, CV_8UC3);
    cv::randu(frame, cv::Scalar(0,0,0), cv::Scalar(255,255,255));

    std::cout << "🚀 正在为 Blackwell (SM 120) 架构预热算子..." << std::endl;
    for(int i=0; i<10; ++i) {
        predictor.predict(frame);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; ++i) {
        predictor.predict(frame);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / 100.0;
    std::cout << "\n========================================" << std::endl;
    std::cout << "🔥 RTX 5060 Ti 推理成功！" << std::endl;
    std::cout << "单帧平均耗时: " << avg_ms << " ms" << std::endl;
    std::cout << "推理吞吐量: " << 1000.0 / avg_ms << " FPS" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
