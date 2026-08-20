---
title: 3.3 The CUDA Driver API
description: CUDA context、module、function 与 Driver API 生命周期
---

# 3.3 The CUDA Driver API

Driver API 比 Runtime API 更直接地暴露 context、module、function、memory 和加载流程。典型生命周期是：初始化 driver，选择设备并创建 context，加载 cubin/PTX/fatbin module，获取 function，配置参数并 launch，再销毁 module/context。

## Context

context 类似 CPU process，Driver API 的资源和动作都封装在 context 中。混用 Runtime API 与 Driver API 时，必须理解当前 context、stream、内存和错误状态的关联；不要把一套 API 分配的对象交给不匹配的路径释放。

## 模块与入口

module 可以来自 cubin、PTX 或 fatbin。加载时可能发生 JIT、符号解析和资源检查；生产环境应记录加载失败、目标 compute capability 和 driver 版本。

需要更高层的设备/内存管理时优先 Runtime API；需要细粒度加载、上下文和模块控制时再使用 Driver API。
