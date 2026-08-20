---
title: 4.19 CUDA Interoperability with APIs
description: CUDA 与其他 API 的互操作
---

# 4.19 CUDA Interoperability with APIs

CUDA 可以与图形、视频、通信或其他计算 API 共享资源和同步。互操作的核心是资源格式、所有权、可见性和同步对象；同一地址可以被两个 API 看到，不代表访问天然安全或有序。
