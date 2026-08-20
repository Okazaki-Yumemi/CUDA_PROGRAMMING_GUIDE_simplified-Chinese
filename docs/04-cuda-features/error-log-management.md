---
title: 4.8 Error Log Management
description: CUDA 错误日志管理
---

# 4.8 Error Log Management

错误日志管理用于收集和定位异步 CUDA 工作中的错误。日志应记录设备、context、stream、调用位置和错误字符串，并区分立即返回的 API 错误、kernel 异步错误和设备状态错误。
