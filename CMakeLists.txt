cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(resnet_inference)

# 1. 路径配置 (使用你 find 到的 Conda 路径)
set(CMAKE_PREFIX_PATH "/root/miniconda/envs/pytorch/lib/python3.9/site-packages/torch")
set(Torch_DIR "${CMAKE_PREFIX_PATH}/share/cmake/Torch")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# 🚀 核心修复：强制使用 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

add_executable(resnet_infer main.cpp)
target_include_directories(resnet_infer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(resnet_infer "${TORCH_LIBRARIES}" ${OpenCV_LIBS})

if(NOT MSVC)
  target_compile_options(resnet_infer PRIVATE -O3 -march=native)
endif()
