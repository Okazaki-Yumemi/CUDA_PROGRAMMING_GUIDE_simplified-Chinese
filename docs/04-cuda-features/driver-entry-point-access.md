---
title: 4.20 Driver Entry Point Access
description: Driver API 入口点访问
---

# 4.20 Driver Entry Point Access

Driver entry point access 提供对驱动入口点获取和版本控制的细粒度方式，便于库根据 driver 能力选择实现。要处理入口点不存在、版本不匹配和可选功能回退，不能把入口点存在当成所有设备行为相同。
