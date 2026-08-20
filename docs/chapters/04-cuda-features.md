---
title: 第四部分：CUDA 功能
description: CUDA Programming Guide Release 13.3 的功能主题中文索引
---

# 第四部分：CUDA 功能

> 原文部分：**4. CUDA Features**。本页保留官方功能名，给出中文标题、问题域和阅读提示。功能支持与 API 细节会随 Toolkit、驱动和 GPU 架构变化。

4.1–4.20 的具体页面已按 NVIDIA 官方网页逐节翻译；本页仍作为功能选择地图，进入各条目后可查看完整代码、表格、注意事项和 API 说明。

## 4.1 Unified Memory：统一内存

统一内存提供统一虚拟地址空间，并通过迁移、预取和一致性机制管理 CPU/GPU 对数据的访问。它降低了显式指针和复制管理的复杂度，但不消除页面迁移、同步和带宽成本。

## 4.2 CUDA Graphs：CUDA 图

CUDA Graph 把 kernel、数据复制、事件和依赖关系组织成可实例化、可重复提交的执行图。对于重复的小任务，它可以减少 host 端提交开销；图的定义、实例化、更新和执行是不同阶段。

## 4.3 Stream-Ordered Memory Allocator：按 stream 顺序的内存分配器

stream-ordered allocator 让分配和释放成为 stream 中有序的异步操作，并可配合内存池减少频繁分配的开销。使用时要确保内存的使用区间被正确的 stream 顺序和事件依赖覆盖。

## 4.4 Cooperative Groups：协作组

Cooperative Groups 用显式组对象表达线程协作、分区和同步。它可以把 warp、block、tile、cluster 等不同作用域写得更清晰，减少把“所有线程”与“当前组”混为一谈的错误。

## 4.5 Programmatic Dependent Launch and Synchronization：程序化依赖启动与同步

该机制允许依赖的 secondary kernel 在 primary kernel 完全结束前按程序化依赖关系启动，从而减少不必要的空等。正确使用需要明确 primary/secondary 的数据依赖和内存可见性边界。

## 4.6 Green Contexts：绿色上下文

Green Context 是与一组特定 GPU 资源关联的轻量上下文，当前重点是 SM 和工作队列资源的划分。它可以让延迟敏感工作获得更可预测的资源，但静态分区可能让其他任务失去可用 SM，需要通过 timeline 评估收益。

## 4.7 Lazy Loading：惰性加载

惰性加载把模块或 kernel 的部分加载工作推迟到第一次实际使用时，以降低启动时开销。代价可能是首次调用延迟和更晚暴露的加载错误；对延迟敏感服务应预热或显式控制加载时机。

## 4.8 Error Log Management：错误日志管理

错误日志管理帮助程序收集、记录和定位异步 CUDA 工作中的错误。错误处理应区分 API 立即返回的错误、kernel 异步错误和设备状态错误，并在日志中记录调用点、stream、设备和上下文。

## 4.9 Asynchronous Barriers：异步 barrier

异步 barrier 允许线程到达、等待和推进阶段分离，适合把数据搬运与计算组织成流水线。使用时要保证每个阶段的参与者数量、到达次数和等待顺序一致，否则容易造成永久等待。

## 4.10 Pipelines：流水线

pipeline 把异步复制、提交、等待和计算分成可重复阶段。双缓冲或多缓冲的基本模式是：当前 tile 计算时，下一 tile 搬运；阶段切换用明确的 barrier 或 pipeline 操作保护。

## 4.11 Asynchronous Data Copies：异步数据复制

异步数据复制让 host/device、global/shared 或其他支持的内存路径与 kernel 执行重叠。可用路径、对齐、粒度、同步和硬件加速能力取决于目标架构，不能只看 API 名称判断实际是否异步。

## 4.12 Work Stealing with Cluster Launch Control：使用 Cluster Launch Control 的工作窃取

Cluster Launch Control 允许在特定 cluster 执行模型中动态获取尚未处理的工作，以改善不均匀工作负载的负载平衡。它适合工作量差异较大的任务，但需要重新检查 block/cluster 的协作假设和剩余工作计数。

## 4.13 L2 Cache Control：L2 缓存控制

L2 cache control 允许程序为特定访问提供持久化、优先级或窗口类提示。提示不是硬保证，实际效果取决于容量、竞争、访问模式和架构；应把它作为测量驱动的优化，而不是正确性依赖。

## 4.14 Memory Synchronization Domains：内存同步域

内存同步域让程序把某些同步/栅栏流量限制在更合适的域中，减少无关工作之间的同步干扰。使用时要先理解 system、device、cluster 或其他作用域的关系，不能用更小域替代真实的数据可见性需求。

## 4.15 Interprocess Communication：进程间通信

CUDA IPC 允许不同进程共享 GPU memory、event 或其他可共享资源。跨进程共享时必须管理句柄生命周期、进程退出、设备可见性和同步；它不是自动解决多进程数据竞争的消息队列。

## 4.16 Virtual Memory Management：虚拟内存管理

VMM 把虚拟地址保留、物理内存分配、映射和访问权限设置拆成可控步骤。它适合构建稀疏/分段地址空间、共享映射或自定义内存管理器，但复杂度明显高于普通 allocation。可参考[图 55：VMM 使用概览](../figures.html#图-55-vmm-使用概览)。

## 4.17 Extended GPU Memory：扩展 GPU 内存

Extended GPU Memory 相关能力面向超出传统设备显存分配模型的内存资源和地址空间使用。具体容量、访问方式、互连与一致性语义需要结合目标 GPU 和 Toolkit 版本确认，不能把它与统一内存直接等同。

## 4.18 CUDA Dynamic Parallelism：CUDA 动态并行

动态并行允许 device 代码配置并启动 child grid。它适合工作递归、数据依赖在 device 端才确定的算法，但会引入 device-side launch、嵌套同步和资源管理开销。要明确 parent/child grid 的关系以及何时可以读取 child 的结果。

## 4.19 CUDA Interoperability with APIs：CUDA 与其他 API 的互操作

CUDA 可以与图形、视频、通信和其他计算 API 共享资源或同步。互操作的核心是明确资源格式、所有权、可见性和同步对象；仅仅把同一块内存地址传给两个 API 通常是不够的。

## 4.20 Driver Entry Point Access：Driver 入口点访问

Driver entry point access 提供对驱动 API 入口点获取和版本控制的更细粒度方式，便于库在不同驱动能力下选择实现。使用时要处理入口点不存在、版本不匹配和可选功能的回退。

## 功能选择速查

| 目标 | 先看 |
| --- | --- |
| 简化 host/device 指针管理 | Unified Memory |
| 重复执行固定工作图 | CUDA Graphs、Lazy Loading |
| 复制与计算重叠 | Asynchronous Data Copies、Pipelines、Asynchronous Barriers |
| 线程组协作 | Cooperative Groups、Thread Block Clusters |
| 降低分配开销 | Stream-Ordered Memory Allocator |
| 资源隔离和尾延迟 | Green Contexts |
| 自定义地址空间 | Virtual Memory Management、Extended GPU Memory |
| 多进程共享资源 | Interprocess Communication |
| device 端生成工作 | CUDA Dynamic Parallelism、Programmatic Dependent Launch |

以上是导航，不是兼容性承诺。真正实施前请确认目标设备的 compute capability、CUDA Toolkit、driver、操作系统、互连和对应 API 的限制。
