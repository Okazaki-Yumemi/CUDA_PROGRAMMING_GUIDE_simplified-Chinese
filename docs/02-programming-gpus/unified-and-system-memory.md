---
title: 2.6 Unified and System Memory
description: 统一内存、系统内存、迁移、预取与可移植性
---

# 2.6 Unified and System Memory

## Unified Memory

统一内存提供统一地址空间，让 CPU 与 GPU 可以访问同一分配；运行时或硬件会在需要时迁移或安排数据。它简化了 host/device 指针管理，但不意味着没有复制或迁移成本。

性能分析应关注页面是否在 CPU/GPU 之间抖动、是否存在并发访问、GPU 是否支持 concurrent managed access、互连带宽以及预取/访问提示是否有效。

## Explicit Memory Management

显式路径使用 `cudaMalloc`、`cudaMemcpy`、`cudaMemcpyAsync` 等 API 控制分配和数据移动。它更适合需要精确控制数据驻留、复制时机、多 GPU 所有权或通信拓扑的程序。

## System and Mapped Memory

操作系统、WSL、Tegra、PCIe、NVLink C2C 和 GPU 架构会影响统一内存和系统内存的支持模式。mapped host memory 可以让 GPU 直接访问 CPU 内存，但访问走 PCIe/NVLink 时延迟和带宽可能使它不适合作为普通 device memory 的替代品。

## 选择策略

先用统一内存建立正确性和数据结构，再用 profiler 决定是否迁移到显式复制；生产程序必须在真实部署硬件上验证，而不是只依据开发机结果。
