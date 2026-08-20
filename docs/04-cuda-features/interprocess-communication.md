---
title: 4.15 Interprocess Communication
description: CUDA 进程间通信
---

# 4.15 Interprocess Communication

CUDA IPC 可以在不同进程之间共享 GPU memory、event 等资源。句柄生命周期、进程退出、设备可见性和同步都要由应用管理；IPC 只提供共享机制，不自动消除数据竞争。
