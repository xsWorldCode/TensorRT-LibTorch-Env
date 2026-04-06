#ifndef INFERENCE_ENGINE_HPP
#define INFERENCE_ENGINE_HPP

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>

class ResNetPredictor {
public:
    ResNetPredictor(const std::string& model_path) {
        try {
            module = torch::jit::load(model_path);
            module.to(torch::kCUDA);
            module.eval();
            std::cout << "✅ LibTorch 加载成功" << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "❌ 失败: " << e.msg() << std::endl;
        }
    }

    int predict(cv::Mat& frame) {
        torch::NoGradGuard no_grad;

        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(224, 224));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

        // 💡 优化点：直接在 CPU 上完成类型转换和归一化，减少 GPU 内部的 Copy 算子调用
        // 这能极大程度避开 "no kernel image" 的基础算子报错
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

        // 一次性搬运到 GPU，不给它做二次转换的机会
        torch::Tensor tensor_in = torch::from_blob(float_img.data, {1, 224, 224, 3}, torch::kFloat32);
        tensor_in = tensor_in.to(torch::kCUDA).permute({0, 3, 1, 2});

        at::Tensor output = module.forward({tensor_in}).toTensor();
        return output.argmax(1).item<int>();
    }

private:
    torch::jit::script::Module module;
};
#endif
