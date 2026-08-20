---
title: 4.3 Stream-Ordered Memory Allocator
description: 按 stream 顺序的内存分配与内存池
---

# 4.3 Stream-Ordered Memory Allocator

按 stream 顺序的分配器把分配/释放放入异步工作队列，并可配合 memory pool 减少频繁分配的开销。释放发生后，其他 stream 仍需通过明确依赖确保不再使用这块内存。

内存池大小、复用、释放阈值和跨 stream 依赖都应在应用生命周期中统一管理。
