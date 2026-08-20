---
title: 第三部分：高级 CUDA
description: 高级 CUDA API、kernel 编程、Driver API、多 GPU 与功能导览
---

# 第三部分：高级 CUDA

> 原文部分：**3. Advanced CUDA**。这一部分更像从“能写 kernel”走向“设计可扩展 CUDA 系统”的导航，很多内容会在第四部分和技术附录中展开。

本部分的 3.1–3.5 已完成按 NVIDIA 官方网页的逐节中文翻译：可从[3.1 高级 CUDA API](../03-advanced-cuda/advanced-apis-and-features.html)开始，也可以直接进入[3.2 高级 Kernel 编程](../03-advanced-cuda/advanced-kernel-programming.html)、[3.3 Driver API](../03-advanced-cuda/driver-api.html)、[3.4 多 GPU](../03-advanced-cuda/multiple-gpus.html)或[3.5 CUDA 功能概览](../03-advanced-cuda/tour-of-cuda-features.html)。

## 3.1 高级 CUDA API 与功能

高级 API 主题通常围绕四类问题展开：如何减少提交开销，如何让多个线程/线程块安全协作，如何管理更复杂的内存与执行资源，以及如何把异步工作组织成可复用的图。

常见选择关系如下：

| 问题 | 可能使用的机制 | 关键风险 |
| --- | --- | --- |
| 重复提交许多小任务 | CUDA Graph、stream capture | 图实例化和更新成本、捕获限制 |
| 线程块之间需要协作 | Cooperative Groups、cluster | 硬件能力、同步作用域、资源占用 |
| 复制与计算重叠 | streams、events、async copy、pipeline | buffer 生命周期和依赖边 |
| 需要更细粒度的地址控制 | Virtual Memory Management | 保留、分配、映射和权限是不同阶段 |
| 需要隔离 GPU 资源 | Green Contexts | 资源划分可能改善延迟但降低弹性 |

这些功能不是“高级版的默认开关”。应先用简单的 kernel 与明确同步建立正确性基线，再一次只引入一种机制，并用 Nsight Systems/Compute 等工具确认它确实改善目标指标。

## 3.2 高级 Kernel 编程

### 3.2.1 让执行配置服务于算法

高级 kernel 设计需要同时考虑 block/grid/cluster 形状、寄存器和 shared memory 用量、warp 访问模式、内存事务与同步。一个能够运行的配置不一定是可移植的最佳配置；把 block size 写死在代码里，也不一定能跨架构保持性能。

### 3.2.2 线程块集群与协作

支持 cluster 的设备可以让一组线程块在单一 GPC 内共同推进，并使用 distributed shared memory。适合的场景包括跨 block 的 tile 协作、需要更大共享工作集的矩阵算法，以及能从 cluster 级同步获益的流水线。使用时必须查询最大 cluster size，并处理设备不支持时的回退路径。

### 3.2.3 内存一致性与同步

同步原语的作用域必须和数据依赖的作用域匹配：warp 内、block 内、cluster 内、device 内或 system 内。把一个较小作用域的 barrier 当成全局 fence，会导致数据竞争。高级代码还要区分“线程已经到达同步点”和“之前的写入对目标线程可见”这两个概念，必要时使用对应的 memory fence/atomic 语义。

### 3.2.4 性能与可观测性

调优应从测量开始：确认 kernel 的总时间、有效带宽、内存事务、分支效率、occupancy、指令吞吐和同步等待。优化一项指标后，重新观察端到端时间，因为提高一个 kernel 的吞吐可能把瓶颈转移到数据准备、host launch 或另一个阶段。

## 3.3 CUDA Driver API

Runtime API 适合大多数应用；Driver API 则提供对 context、module、function、memory 和加载过程更直接的控制。常见 Driver API 生命周期是：

1. 初始化 CUDA driver；
2. 枚举设备并创建或选择 context；
3. 加载 cubin/PTX/fatbin 中的 module；
4. 获取 kernel function 或 global symbol；
5. 配置参数并 launch；
6. 通过 stream/event 管理异步工作；
7. 销毁 module、context 和相关资源。

context 类似 CPU 进程，driver API 中的资源和动作都属于某个 context。混用 Runtime API 和 Driver API 时，要理解二者如何关联当前 context、stream 和错误状态；不要把一个 API 分配的资源随意交给另一套 API 释放。

## 3.4 多 GPU 系统编程

多 GPU 程序首先要回答：数据由哪张 GPU 持有？哪些 kernel 在哪张 GPU 执行？设备之间如何复制或通信？GPU 之间的 PCIe、NVLink、P2P 能力和拓扑都会影响答案。

### 3.4.1 设备选择

启动时枚举设备属性，选择计算能力、显存容量、SM 数量和互连条件满足要求的设备。不要用设备编号作为业务语义；设备枚举顺序可能由环境和配置改变。将“物理 GPU”“逻辑设备”“当前 context”分别记录，避免排查时混淆。

### 3.4.2 数据与工作划分

常见划分方式有：按数据切分、按 pipeline 阶段切分，或让每张 GPU 处理独立请求。数据切分需要考虑边界交换、结果归并和 load balance；独立请求需要考虑每张 GPU 的队列、上下文和内存池。多个 GPU 同时访问同一 host buffer 时，必须明确同步和所有权。

### 3.4.3 P2P 与通信

如果设备支持 peer access，可以避免部分经过 host memory 的往返；但 P2P 是否可用取决于设备对、拓扑、驱动和系统配置。跨 GPU 的 copy 仍然是异步工作，使用 event 和 stream 表达依赖，并用 profiler 检查链路是否成为瓶颈。

## 3.5 CUDA 功能导览

本节是第四部分的地图。第四部分按功能主题详细展开：统一内存、CUDA Graphs、stream-ordered allocator、Cooperative Groups、程序化依赖启动、Green Contexts、Lazy Loading、错误日志、barrier、pipeline、异步数据复制、cluster launch control、L2 cache control、memory synchronization domains、IPC、VMM、extended GPU memory、dynamic parallelism、API interoperability 和 driver entry point access。

对高级 CUDA 的稳妥使用顺序是：

1. 先用普通 stream 与显式同步写出正确版本；
2. 用测量定位 host launch、同步、内存搬运或 kernel 本身的瓶颈；
3. 选择能直接针对瓶颈的功能；
4. 添加设备能力查询和回退路径；
5. 在目标 GPU、驱动和操作系统组合上验证，不只在开发机上验证。

## 本部分小结

高级 CUDA 的难点不是 API 数量多，而是作用域、生命周期、异步依赖和硬件能力同时参与正确性。把每个功能都记录为“资源是什么、作用域是什么、何时可见、如何失败、如何回退”，比单纯记住函数名更可靠。

继续阅读[第四部分：CUDA 功能](./04-cuda-features.html)或回到[NVIDIA 官方目录](https://docs.nvidia.com/cuda/cuda-programming-guide/contents.html)核对最新 API 页面。
