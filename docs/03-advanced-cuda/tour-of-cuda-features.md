---
title: 3.5 A Tour of CUDA Features
description: CUDA 功能的主题导航
---

# 3.5 A Tour of CUDA Features

这一节把 CUDA 能力按问题域串起来：统一内存简化数据共享，CUDA Graph 减少重复提交开销，Cooperative Groups 和 cluster 明确线程协作，异步 copy/pipeline 重叠搬运与计算，VMM 和扩展内存提供更细粒度的地址控制，多 GPU/IPC/互操作扩展系统边界。

## 按问题检索

- 重复的工作流：CUDA Graphs、stream-ordered allocator、lazy loading；
- 线程组协作：Cooperative Groups、cluster、asynchronous barriers；
- 复制和流水线：asynchronous data copies、pipelines、TMA；
- 地址与资源：VMM、extended GPU memory、Green Context；
- 系统集成：IPC、API interoperability、Driver entry point access；
- device 端生成工作：dynamic parallelism、programmatic dependent launch。

功能页面会说明使用场景，但 API 支持和硬件限制仍以当前 NVIDIA 官方文档为准。
