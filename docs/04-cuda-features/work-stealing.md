---
title: 4.12 Work Stealing with Cluster Launch Control
description: Cluster Launch Control 工作窃取
---

# 4.12 Work Stealing with Cluster Launch Control

Cluster Launch Control 允许特定 cluster 执行模型动态获取尚未处理的工作，用于改善工作量不均匀时的负载平衡。它需要重新检查剩余工作计数、block/cluster 协作和目标设备能力。
