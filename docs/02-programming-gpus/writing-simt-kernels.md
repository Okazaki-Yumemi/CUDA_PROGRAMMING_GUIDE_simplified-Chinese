---
title: 2.3 Writing SIMT Kernels
description: SIMT kernel、线程层次、内存空间、访问模式和性能
---

# 2.3 Writing SIMT Kernels

## Thread Hierarchy

先确定一个工作项对应一个线程、多个元素还是一个 tile 的部分，再设计 block/grid 映射。block 是同步和 shared memory 的主要作用域，grid 是总体任务规模；block 间没有隐式顺序。

## GPU Device Memory Spaces

寄存器适合线程私有变量；shared memory 适合 block/cluster 内协作；global memory 容量大但要关注事务和复用；constant/read-only/texture cache 适合特定只读模式；local memory 名字像片上内存，但大局部数组或寄存器溢出可能把它放到较慢的 device memory。

## Coalesced Global Memory Access

一个 warp 的 lane 如果访问连续且对齐的 global addresses，硬件通常可以用较少事务满足请求。分析 kernel 时应画出 lane 到地址的映射，分别检查 load 和 store；转置和跨步访问常用 shared memory 进行局部重排。

## Shared Memory Access Patterns

shared memory 的 bank 映射会影响并行服务。步长访问、转置和不规则索引可能产生 bank conflict；`32 x 32` tile 增加一列 padding 是常见的布局技巧。可参考[图 15](../figures.html#图-15-共享内存中的步长访问)和[图 18](../figures.html#图-18-带-padding-的共享内存)。

## Atomics, Cooperative Groups and Occupancy

原子操作保护单个位置的更新，但热点会串行化。Cooperative Groups 让 warp、block、tile、cluster 等协作范围更明确。occupancy 只是驻留线程数相对上限的比例；高 occupancy 不自动等于高性能，必须结合内存延迟、指令吞吐、寄存器和同步分析。

## 性能检查清单

1. 边界和数据竞争是否正确；
2. warp 是否大量分歧；
3. global memory 是否合并、对齐；
4. shared memory 是否 bank conflict；
5. 寄存器/shared memory 用量是否限制 block 驻留；
6. profiler 中的理论分析是否与实际事务、吞吐和等待相符。
