---
title: 3.2 Advanced Kernel Programming
description: 高级 kernel 的配置、协作、内存一致性和性能
---

# 3.2 Advanced Kernel Programming

高级 kernel 同时受 block/grid/cluster 形状、寄存器、shared memory、warp 访问、内存事务与同步影响。可运行的配置不一定是可移植的最佳配置；应把资源用量、目标架构和回退策略一并设计。

## 协作与一致性

block 内使用 shared memory 和 block barrier；cluster 内使用 cluster 级协作和 distributed shared memory；grid/device/system 级同步则需要更明确的 API 或分阶段 kernel。同步点既要保证线程到达，也要保证需要的数据在目标作用域可见。

## 性能

观察 kernel 时间、有效带宽、内存事务、分支效率、occupancy、指令吞吐和同步等待。高 occupancy 不自动等于高性能；寄存器压缩、shared memory 过度使用或同步减少也可能带来新的瓶颈。
