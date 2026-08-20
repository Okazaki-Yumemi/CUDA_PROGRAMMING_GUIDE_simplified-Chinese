---
title: 3.1 Advanced CUDA APIs and Features
description: 高级 CUDA API 的选型、作用域和生命周期
---

# 3.1 Advanced CUDA APIs and Features

高级 CUDA API 主要解决提交开销、线程组协作、异步流水线、复杂内存和资源隔离问题。常用选择包括 CUDA Graph、Cooperative Groups、cluster、async copy、pipeline、VMM 和 Green Context。

## 先建立基线

先用普通 stream、明确的内存管理和同步写出正确版本；再用 profiler 判断瓶颈来自 host launch、复制、同步、内存带宽还是 kernel。一次只引入一种高级机制，并保留不支持目标设备时的回退路径。

## 作用域与生命周期

每个机制都要回答：资源属于哪个 context/stream/线程组？何时创建？何时可见？谁负责释放？失败时怎么回退？把较小作用域的 barrier 当作全局同步，是高级 CUDA 中最常见的错误之一。

继续查看[4. CUDA Features](../04-cuda-features/unified-memory.html)的逐功能页面。
