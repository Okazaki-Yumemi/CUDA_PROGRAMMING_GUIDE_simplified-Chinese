---
title: 5.7 CUDA C++ Memory Model
description: CUDA C++ 内存模型、原子性、同步和数据竞争
---

# 5.7 CUDA C++ Memory Model

本节解释线程作用域、同步原语、原子性、数据竞争和消息传递。判断并发代码是否正确时，必须同时问：写入是否原子、何时可见、哪个作用域需要同步、读取是否可能发生在发布之前。
