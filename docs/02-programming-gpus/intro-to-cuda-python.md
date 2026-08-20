---
title: 2.2 Intro to CUDA Python
description: CUDA Python 生态、kernel、数组和同步
---

# 2.2 Intro to CUDA Python

CUDA Python 是一组生态路径，而不是单一语言实现。Numba 可以 JIT 编译 CUDA kernel；CuPy 提供 GPU array；PyTorch、JAX 等框架提供 tensor、stream 和库调用；其他封装则把 CUDA libraries 暴露给 Python。

## CUDA Python Ecosystem

选择路径时先看现有数据结构：已有 NumPy 风格数组时，数组库通常最快；已有深度学习框架时，优先复用框架的 CUDA tensor 和 stream；只有在库无法表达需求时，才需要自定义 Python kernel。

## SIMT Kernels in Python

Numba 风格的向量加法如下：

```python
from numba import cuda

@cuda.jit
def vector_add(a, b, c):
    i = cuda.grid(1)
    if i < c.size:
        c[i] = a[i] + b[i]

vector_add[blocks, threads](a, b, c)
cuda.synchronize()
```

Python 语法更短，但 thread/block 索引、边界检查、内存驻留、stream 和异步错误仍然是 CUDA 语义的一部分。GPU array 的 dtype、layout 和设备选择也会影响性能。

## Running CUDA Python Applications

建议把程序分为 setup、数据准备、GPU 工作、同步/检查和清理几个阶段。不要因为 Python 函数已经返回，就假定 GPU 已经完成；在读取 host 结果或关闭资源前使用库提供的同步接口。

## 进一步阅读

SIMT 性能、内存空间和错误检查见[2.3 Writing SIMT Kernels](./writing-simt-kernels.html)；统一内存和 host/device 数据移动见[2.6 Unified and System Memory](./unified-and-system-memory.html)。
