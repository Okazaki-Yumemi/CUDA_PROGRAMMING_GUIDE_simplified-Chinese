---
title: 5.8 CUDA C++ Execution model
description: CUDA C++ host/device 执行模型
---

# 5.8 CUDA C++ Execution model

本节描述 host thread、device thread、CUDA API、依赖和执行环境之间的关系。它适合解释 kernel launch 为何异步、不同 stream 为何可能重叠，以及同步和错误检查为何要放在正确的位置。
