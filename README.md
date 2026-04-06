# TensorRT-LibTorch-Env (ResNet-54 推理部署)

本项目是一个基于 **LibTorch** 和 **TensorRT** 的工业级视觉推理框架，针对 **NVIDIA RTX 5060 Ti (Blackwell 架构)** 进行了优化。

## 🚀 项目特性
* **高性能架构**：支持 ResNet-54 模型的分类与推理。
* **硬件优化**：适配 sm_120 算力核心，利用 CUDA 12.8 加速。
* **双模部署**：提供 C++ 源码及预编译的二进制执行文件。

## 📂 仓库结构
* `main.cpp` / `InferenceEngine.hpp`: 核心 C++ 推理逻辑。
* `models/model.pt`: 预训练的 ResNet-54 权重文件。
* `build/resnet_infer`: 针对 50-series 显卡编译的可执行文件。
* `CMakeLists.txt`: 自动化构建脚本。

## 🛠️ 环境要求
* **OS**: Ubuntu 22.04 / WSL2
* **GPU**: NVIDIA RTX 5060 Ti (或其它 Blackwell 架构显卡)
* **CUDA**: 12.8+
* **LibTorch**: 2.x (Pre-built with CUDA)

## 🏃 快速开始

### 方式 1：直接运行 (推荐)
如果你拥有相同的硬件环境，可以直接运行已提交的二进制文件：
```bash
cd build
./resnet_infer
