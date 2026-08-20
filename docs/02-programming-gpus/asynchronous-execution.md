---
title: 2.5 Asynchronous Execution
description: CUDA stream、event、graph、异步复制与流水线
---

# 2.5 Asynchronous Execution

CUDA 的异步接口让 host 提交 kernel、复制或其他工作后立即返回。异步不等于工作一定并行；是否重叠取决于 stream 依赖、硬件复制引擎、资源和数据生命周期。

## Streams and Events

同一个 stream 内的操作有序；不同 stream 之间需要用 event 或其他依赖表达先后。event 可以记录某个 stream 的时间点，再让另一个 stream 等待它。

## CUDA Graphs

Graph 把 kernel、复制、event 和依赖关系组织为可实例化的执行图。定义、instantiate 和 execute 是不同阶段。大量重复的小任务可以减少 host 提交开销，但构图和更新成本也必须测量。

## Async Copies and Pipelines

双缓冲流水线通常让当前 buffer 计算、下一 buffer 搬运；切换时用 barrier、event 或 pipeline API 保证数据已经可读。`cudaMemcpyAsync`、shared memory async pipeline 和 TMA 对对齐、支持硬件、同步语义有各自要求。

## 常见错误

- 把不同 stream 的提交顺序误当成依赖；
- buffer 还在 GPU 使用时被 host 复用；
- 把 `cudaDeviceSynchronize()` 当成唯一优化方式，导致 CPU/GPU 被迫串行；
- 只测 kernel 时间，不测复制、提交和同步等待。
