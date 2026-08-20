---
title: '2.7 NVCC: The NVIDIA CUDA Compiler'
description: nvcc 的 host/device 编译流程和架构代码生成
---

# 2.7 NVCC: The NVIDIA CUDA Compiler

NVCC 协调 host 编译器与 device 编译流程：识别 `.cu` 文件、处理 CUDA 语言扩展、生成 PTX/cubin/fatbin，并把 host/device 代码链接成可执行程序或库。

## 构建检查

- `-arch` / `-gencode` 是否覆盖部署 GPU；
- 是否要保留 PTX 以支持未来设备或 JIT；
- host 编译器版本、C++ 标准和 ABI 是否匹配；
- debug 构建是否保留 device 行号和调试信息；
- release 构建是否检查寄存器、shared memory、代码大小和启动配置；
- device linking、模板、Lambda、宏和 host/device 函数声明是否符合当前 Toolkit 支持。

## Host 与 Device 函数

`__host__`、`__device__`、`__global__` 表达函数在哪一侧编译/调用。host code 可以 launch `__global__` kernel；device code 不能任意调用 host-only API。相同函数同时编译为 host/device 版本时，也要确认其依赖和语言特性在两侧都受支持。

## 架构与发布

可执行文件中的 fatbin 可以包含多个 cubin 和 PTX。为特定架构提供 cubin 可减少首次 JIT 延迟，保留 PTX 可以扩大兼容范围，但会增加包体积和首次启动工作。发布策略要根据实际支持设备、driver 和 Toolkit 版本重新评估。

更多硬件限制见[第五部分：技术附录](../chapters/05-technical-appendices.html)，更完整的 CUDA C++ 代码见[2.1 Intro to CUDA C++](./intro-to-cuda-cpp.html)。
