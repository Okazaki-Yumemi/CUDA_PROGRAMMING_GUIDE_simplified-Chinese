---
title: 1.3 The CUDA platform
description: CUDA 平台、Toolkit、Driver、PTX、cubin 与 fatbin 的中文翻译
---

# 1.3 The CUDA platform

## Compute Capability 与 Streaming Multiprocessor 版本

CUDA 用 **compute capability** 描述一组 GPU 硬件能力。它影响可用指令、内存空间、原子操作、cluster、异步数据搬运、资源上限和其他功能。compute capability 不是简单的性能等级；同样的 kernel 可能在不同能力级别上拥有不同的功能路径和限制。

程序可以查询设备属性，根据目标能力选择 kernel launch 配置或启用可选功能。发布程序时也要决定要提供哪些架构的 cubin 与 PTX。

## CUDA Toolkit 与 NVIDIA Driver

CUDA Toolkit 提供编译器、头文件、库、调试器和分析工具；NVIDIA Driver 让用户程序与 GPU 交互。CUDA Runtime API 为应用提供较高层的设备、内存、stream、event 和 kernel 管理；CUDA Driver API 暴露 context、module、function 等更底层资源。

Runtime API 与 Driver API 可以互操作，但要明确当前 context、资源所有权和错误状态。`cudaMalloc`、`cudaStreamCreate` 等 runtime 操作不应和不匹配的 driver 释放路径混用。

## Parallel Thread Execution（PTX）

PTX 是 CUDA 平台的虚拟指令集和中间表示。高层 CUDA 代码可以编译到 PTX，驱动在目标 GPU 上做即时编译（JIT）；也可以在构建时生成针对具体架构的 cubin。

PTX 帮助程序跨代 GPU 保持一定的可移植性，但不保证新架构上性能自动相同。PTX 是虚拟 ISA，不等同于某一颗 GPU 的最终机器码。

## Cubins 与 Fatbins

cubin 是针对具体 GPU 架构的二进制代码；fatbin 是可容纳多个 GPU 代码变体的容器。可执行文件或库通常包含 CPU 二进制和 GPU fatbin，fatbin 可以同时包含多个 cubin 与 PTX。

运行时会选择合适的 cubin；如果缺少可直接执行的 cubin，则可能使用 PTX JIT。兼容性因此有两个维度：已经生成的二进制是否能在目标设备执行，以及目标驱动能否把 PTX 编译到目标设备。

![PDF 第 33 页的图 10：fatbin、cubin 与 PTX](/figures/page-033.png)

*PDF p.33，图 10。CPU 代码和 GPU fatbin 位于同一可执行文件/库中；fatbin 可以携带 cubin 和 PTX。*

## 构建时的实用决策

- 需要最低首次启动延迟：为目标架构提供匹配的 cubin；
- 需要覆盖未来/未知架构：保留合适的 PTX，但接受 JIT 可能带来的首次延迟；
- 需要控制包体积：减少不必要的架构变体；
- 需要支持多个部署环境：结合目标 GPU 列表设置 `-gencode`，并在实际设备上测试；
- 需要诊断加载：记录设备能力、选中的代码变体、JIT 和 driver 错误。

这套打包策略会在 CUDA Toolkit、Driver 和 GPU 架构更新时重新评估。不要只根据旧版博客中的架构列表做长期发布策略。

## 本部分入口

- [1.1 Introduction](./introduction.html)
- [1.2 Programming Model](./programming-model.html)
- [第二部分：用 CUDA 编程 GPU](../02-programming-gpus/intro-to-cuda-cpp.html)
