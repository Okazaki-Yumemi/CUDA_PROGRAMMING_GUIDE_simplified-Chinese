---
title: 2.2 CUDA Python 入门
description: CUDA Python 生态、SIMT kernel、GPU 数组、同步与错误检查
---

# 2.2 CUDA Python 入门（Intro to CUDA Python）

> 本页按 NVIDIA CUDA Programming Guide Release 13.3 的官方网页逐段翻译。原文页面：[2.2. Intro to CUDA Python](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-python.html)；官方页面标注的更新时间为 2026-05-27。

本章介绍使用 Python 编写 CUDA kernel。CUDA Python 生态封装了一个规模庞大、仍在积极演进的工具和库生态。本章先介绍其中一些组件，再使用其中几个组件说明如何在 Python 中编写和执行 GPU 代码。

在 Python 中利用 GPU 计算有许多方式，其中很多方式不要求显式编写 GPU kernel。[CUDA Python 生态](#221-cuda-python-生态-cuda-python-ecosystem)中的一些组件提供了直接在 GPU 上执行操作的函数，开发者不需要进行具体的 GPU 控制或编程。NVIDIA [Accelerated Computing Hub](https://github.com/NVIDIA/accelerated-computing-hub) 提供了 [Accelerated Python User’s Guide](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/Accelerated_Python_User_Guide/notebooks)，介绍并讨论许多支持 Python GPU 加速计算的库和工具。对于希望尽快、尽量容易地在 Python 中使用 GPU、但不一定要直接编写 GPU 代码的用户，这是一条很好的起点。

与此不同，本章关注直接控制 GPU，以及在 Python 中编写并在 GPU 上执行的 kernel。本章重点介绍在 Python 中进行 [CUDA 单指令多线程（SIMT）](../01-introduction/programming-model.html)编程。

## 2.2.1 CUDA Python 生态（CUDA Python Ecosystem）

CUDA Python 是一组用于在 Python 中进行 GPU 计算的工具和库。下面的列表介绍 CUDA Python 的主要组成部分，但其中并非所有组件都是本章内容所必需的。该列表改编自完整的 [CUDA Python GitHub 仓库](https://github.com/NVIDIA/cuda-python)列表。

**主要组件——用于 GPU 控制和运行库提供的 GPU 代码**

- `cuda.core`：用于 CUDA 控制（例如内存和设备管理）的 Python 风格接口，为 Python 提供类似 CUDA C++ Runtime 的功能。
- `cuda.compute`：提供 [CUDA Core Compute Library（CCCL）](https://nvidia.github.io/cccl/unstable/python/compute.html) GPU 加速函数的 Python 模块。
- `CuPy`：提供 GPU 加速版 NumPy 例程，以及 GPU 加速的 `ndarray` 数据容器的 Python 库。

**Kernel 编写组件**

- `cuda.lang`：一种 Python 领域专用语言（DSL），使用 Python 语言的一个子集，在 SIMT 编程模型中编写 CUDA kernel 和 device 函数。
- `cuda.coop`：提供 [CUDA Core Compute Library（CCCL）](https://nvidia.github.io/cccl/unstable/python/coop.html)原语的 Python 模块；这些原语可以由 device 调用，并使用 `cuda.lang` 编写。
- `cuda.tile`：一种 Python DSL，用于在 Tile 编程模型中编写 CUDA kernel 和 device 函数。

**其他组件**

- `cuda.pathfinder`：用于在 Python 环境中定位 CUDA 组件的工具。
- `cuda.bindings`：CUDA 库和工具的底层 Python 绑定，包括 CUDA Driver API、CUDA Runtime API、NVRTC、NVVM 等。`cuda.bindings` 通过 CUDA Driver 和 CUDA Runtime 组件提供与 `cuda.core` 相同的功能，但它们是 C 语言 API 的 Python 包装，而不是原生 Python 风格的接口。

### 2.2.1.1 在 Python 中使用 CUDA 库（Using CUDA Libraries in Python）

CUDA C++ 拥有丰富的库生态，不需要直接编写 kernel 或 GPU 代码就可以获得 GPU 加速。CUDA C++ 于 2006 年推出时，库很少，开发者在很大程度上必须自己编写 GPU kernel。此后，大量[库](https://developer.nvidia.com/cuda/cuda-x-libraries)被开发出来，使开发者几乎无需编写 GPU 代码就能在 C++ 中利用 GPU 计算。

CUDA Python 生态则从另一个方向发展：在能够直接使用 Python 语法和语义编写自定义 kernel 之前，CuPy 等 Python 库已经向 Python 开发者提供了 GPU 加速的计算和算法实现。其中许多库为 CUDA C++ 实现的 GPU 代码提供了 Python 绑定。

在现代 CUDA 中，如果 GPU 加速库能够以足够的表达能力满足需求，几乎总是建议优先使用这些库。许多库的实现经过 GPU 计算专家调优。当没有可用的库，或者现有库不足以表达需求时，也可以像 C++ 一样直接在 Python 中编写 GPU kernel 和 device 函数。

### 2.2.1.2 本章范围（Scope of This Chapter）

虽然开发者应尽可能优先使用库，但本章后面的内容会介绍：当需要编写自定义 Python GPU 代码时应如何进行。本章以与 [2.1 节](./intro-to-cuda-cpp.html)处理 C++ 相同的方式，介绍如何在 GPU 上编写和运行自定义 Python 代码：先从如何指定 GPU kernel 开始，然后介绍如何使用 CuPy 提供的 GPU 加速 `ndarray`，在 GPU 上分配内存并实现 CPU 与 GPU 之间的通信。

### 2.2.1.3 环境准备（Getting Setup）

一般来说，CUDA Python 生态的大多数组件都可以在 PyPI 上获得，可以使用 `pip` 或任何常见的 Python 包管理器安装。所有这些包都要求系统安装最新的 NVIDIA Driver。编写或运行 CUDA Python 应用通常不要求安装 CUDA Toolkit。

不同平台上 CUDA Python 的安装和配置方法见 NVIDIA Developer Zone 的 [CUDA Python](https://developer.nvidia.com/how-to-cuda-python) 页面。

### 2.2.1.4 运行 CUDA Python 应用（Running CUDA Python Applications）

无论使用 CUDA 加速库还是用户编写的 GPU 代码，CUDA Python 应用的运行方式都与普通 Python 应用相同。本节示例都从命令行调用 `python3`，运行名为 `cuda-python-app.py` 的程序：

```bash
$ python3 cuda-python-app.py
```

## 2.2.2 Python 中的 SIMT Kernel（SIMT Kernels in Python）

正如 [CUDA 编程模型](../01-introduction/programming-model.html)的引言所述，能从 host 调用并在 GPU 上执行的函数称为 kernel。CUDA 提供两种模型：[单指令多线程（SIMT）](../01-introduction/programming-model.html)和 [CUDA tile](../01-introduction/programming-model.html)。SIMT kernel 被编写成由许多并行线程同时运行；这一点在 CUDA Python 和 CUDA C++ 中相同。本章使用 SIMT kernel 介绍 CUDA Python。

### 2.2.2.1 指定 Kernel（Specifying Kernels）

在 CUDA Python 中指定 kernel 之前，必须导入 `numba.cuda` 包，通常写成：

```python
from numba import cuda
```

这会导入 `numba.cuda` 包，使程序能够使用该包提供的 `cuda` 命名空间中的组件。

要在 CUDA Python 中把函数指定为 kernel，应在函数定义上一行放置 `@cuda.jit` 装饰器：

```python
from numba import cuda

@cuda.jit
def function(input_array, output_array):
    ...
```

这样做会使 kernel 在第一次启动时针对当前 GPU 执行 JIT 编译。如果没有指定其他 GPU，就会使用默认 CUDA 设备，本节示例就是如此。

### 2.2.2.2 启动 Kernel（Launching Kernels）

执行 kernel 的线程数量在 kernel launch 中指定，这称为执行配置。每次调用 kernel 都可以使用独立的执行配置，例如不同的 block size 或 thread block 数量。

#### 2.2.2.2.1 Kernel Launch

要启动 kernel，应把执行配置放在 kernel 名称之后、函数参数之前的方括号 `[ ]` 中。参数顺序与 C++ 中介绍的三尖括号记法相同，具体形式为：

```text
kernel_name[number_of_thread_blocks, threads_per_block](arguments, ...)
```

下面的代码片段展示了如何在 Python 源文件中定义并调用 kernel：

```python
from numba import cuda

@cuda.jit
def my_kernel(input, output):
    ...

## launch the kernel
my_kernel[num_thread_blocks, threads_per_block](in_array, out_array)
```

每个 block 的线程数量存在上限，因为 block 中的所有线程都驻留在同一个流式多处理器（SM）上，并且必须共享该 SM 的资源。在当前 GPU 上，一个 thread block 最多可以包含 1024 个线程。如果资源允许，一个 SM 可以同时调度多个 thread block。

#### 2.2.2.2.2 多维 Grid 与 Thread Block（Multi-Dimension Grids and Thread Blocks）

在 CUDA 中，thread block 以及由 thread block 组成的 grid 可以是一维、二维或三维。grid 或 thread block 为一维时，可以用一个整数指定 kernel launch 的执行配置。thread block 或 grid 为二维或三维时，则使用二维或三维 tuple。下面展示二维 grid 和二维 thread block 的启动方式，其中 `gridX`、`gridY` 是 grid 的 x、y 维度，`blockX`、`blockY` 是 grid 中每个 thread block 的 x、y 维度。

```python
from numba import cuda

@cuda.jit
def function(input, output):
    ...

## launch the kernel
function[(gridX, gridY), (blockX, blockY)](in_array, out_array)
```

### 2.2.2.3 线程与 Grid 索引内建变量（Thread and Grid Index Intrinsics）

[第 1.2.2.1 节](../01-introduction/programming-model.html)介绍了线程和 grid，[第 2.2.2.2 节](#2222-多维-grid-与-thread-block-multi-dimension-grids-and-thread-blocks)展示了如何指定 kernel launch 的 grid 和 thread block 大小。在 kernel 内部，每个线程都可以访问执行配置参数、自己的线程索引，以及自己所属的 thread block 在 grid 中的索引。

kernel 函数可以访问下面这些变量，以确定线程的身份：

- `cuda.threadIdx.[xyz]`：线程在所属 thread block 内的索引。一个 block 中的每个线程都有不同的索引。
- `cuda.blockDim.[xyz]`：thread block 的维度，即 kernel launch 执行配置中指定的维度。
- `cuda.blockIdx.[xyz]`：thread block 在 grid 内的索引。每个 thread block 都有不同的索引。
- `cuda.gridDim.[xyz]`：grid 的维度，即启动 kernel 时执行配置中指定的维度。

这些变量都是包含 `.x`、`.y`、`.z` 成员的三分量向量。如果 kernel launch 的执行配置没有指定某个维度，则该维度的大小默认为 1，而索引默认为 0。

`cuda.threadIdx` 和 `cuda.blockIdx` 都从 0 开始索引。例如，`cuda.threadIdx.x` 的取值从 0 到 `cuda.blockDim.x - 1`（含两端）；`.y` 和 `.z` 在各自维度中遵循同样规则。

下面给出一个简单的向量加法 kernel，它把两个向量逐元素相加。这个函数接收三个数组 `A`、`B` 和 `C`，实现逐元素的 `C = A + B`：

```python
# C = A + B vector addition
@cuda.jit
def vecadd(A, B, C):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    C[idx] = A[idx] + B[idx]
```

kernel 首先计算线程在 grid 中的唯一索引。这个 kernel 假设使用一维 thread block 和一维 grid 启动。`idx` 的取值从 0 到 `N-1`，其中 `N` 是 grid 中的线程总数，即 `N = cuda.gridDim.x * cuda.blockDim.x`。

上面代码中的线程索引计算模式非常常见，因此 Numba 提供了简写语法 `cuda.grid(n)`，其中 `n` 是维数。在上面的例子中，下面这行代码：

```python
idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
```

可以替换为更简单的：

```python
idx = cuda.grid(1)
```

需要注意的是，这个 kernel 没有检查访问 `A`、`B` 或 `C` 时是否越界。本章假设这些数组是由 CuPy 创建的 `ndarray`，将在稍后的[第 2.2.3.3 节](#2233-ndarray-对象类型-the-ndarray-object-type)介绍。使用 CuPy `ndarray` 时，数组类型会隐式执行边界检查。

## 2.2.3 GPU 计算中的内存（Memory in GPU computing）

::: tip 注意
CuPy 等 Python 包会直接使用 CUDA C++ API 管理 GPU 内存，例如[第 2.1.3.2 节](./intro-to-cuda-cpp.html)介绍的 API。多个 Python 包都提供了用于控制 GPU 内存分配的包装和工具。本指南只介绍 CuPy。多数包的概念相似，除非特别说明，它们的行为也通常与 C++ 对应功能类似。
:::

正如[第 1.2.3 节](../01-introduction/programming-model.html)所述，GPU 带有连接到它的 DRAM。kernel 将使用的数据数组通常需要在从 kernel 访问之前位于 GPU 的 DRAM 中。在 Python 中，控制数据在内存中的位置——即在 CPU 与 GPU 之间移动数据——由程序员负责。这与 C++ 中介绍的显式内存管理情况相同。

### 2.2.3.1 在 GPU 上实例化数组（Instantiating arrays on the GPU）

CuPy 提供了在 GPU 上创建指定类型和维度的 `ndarray` 对象，以及在 CPU 与 GPU 之间复制数据的函数。CuPy 的许多函数与 NumPy 中创建 `ndarray` 的函数具有相似的签名。下面给出使用 CuPy 在 GPU 内存中创建和填充数组的几个例子。

```python
import cupy as cp
import numpy as np

## create a matrix of zeros on the GPU
## when a datatype is not specified, float32 is used by default
A_device = cp.zeros((1024, 1024))

## create an array of 2^20 random doubles on the GPU
B_device = cp.rand.random((2**20), dtype=np.double)

## create an array of zeroes with the same shape and datatype as an existing array
C_device = cp.zeros_like(A)
```

### 2.2.3.2 在 Host 与 GPU 内存之间复制数组（Copying Arrays between the Host and GPU memory）

CuPy 也可以把驻留在 CPU 内存中的 NumPy `ndarray` 复制为驻留在 GPU 内存中的 CuPy 数组。

```python
import cupy as cp
import numpy as np

## Create an array in host memory
A_host = np.zeros((1024, 1024))
## Copy the array to the GPU
A_device = cp.array(A_host)

## Create an array in GPU memory
B_device = cp.rand.random((1024, 1024))
## copy the array to host memory
B_host = cp.asnumpy(B_device)
```

### 2.2.3.3 `ndarray` 对象类型（The ndarray object type）

上一节中的 `ndarray` 对象只存在于 host memory 或 GPU memory 其中之一，而不会同时存在于两者。把驻留在 host 上的数组作为参数传给 kernel 会产生错误；把驻留在 GPU memory 中的数组传给普通 Python 函数（即不是 kernel 的函数）也会产生错误。

CuPy 不会隐式执行 CPU 与 GPU 之间的复制，因为复制可能代价很高，过多的数据复制会损害性能。因此，CuPy 要求程序员明确决定何时在 CPU 与 GPU 之间复制数据。

在 GPU kernel 中使用 `ndarray` 类型的一个优点是，数组自身带有各维度的范围信息。正如[第 2.2.2.3 节](#2223-线程与-grid-索引内建变量-thread-and-grid-index-intrinsics)所示，当所需线程总数略小于执行 block 或 grid 的总大小时，边界检查会自动完成，kernel 代码不需要检查越界访问。

## 2.2.4 同步 CPU 与 GPU（Synchronizing the CPU and the GPU）

和 C++ 一样，CUDA Python 中的 kernel launch 相对于 host 线程也是异步的。也就是说，kernel launch 后 host 代码会继续在 CPU 上执行，而不保证 kernel 已经完成，甚至不保证 kernel 已经开始执行。为了保证 GPU kernel 执行完成，host 线程必须执行某种 GPU 同步。

最简单的同步形式是同步整个 GPU。这种设备范围的同步由 CUDA driver 提供，CuPy 和 `numba.cuda` 都通过 `synchronize()` 方法向 Python 暴露：

```python
import cupy as cp
from numba import cuda

..

## Wait on host thread for all pending GPU work to complete
## this uses the interface provided by cupy
cp.synchronize()

## Wait on host thread for all pending GPU work to complete
## this uses the interface provided by numba.cuda
cuda.synchronize()
```

设备范围同步会让 host 线程等待，直到 GPU 上之前发出的所有工作完成。也可以使用 CUDA stream 执行更细粒度的同步，详见[第 2.5 节：异步执行](./asynchronous-execution.html)。在 Python 中使用 stream 时，建议使用 `cuda.core` 创建 CUDA stream，并根据需要只与特定 stream 同步。

## 2.2.5 综合示例（Putting it All Together）

下面给出最常见的第一个 GPU kernel：执行并行向量加法的 Python 版本。

```python
import numpy as np
from numba import cuda
import cupy as cp


## Defines a CUDA kernel to perform C = A + B vector addition
@cuda.jit
def vecadd(A, B, C):
    work_index = cuda.grid(1)
    C[work_index] = A[work_index] + B[work_index]


# note that vector size is not a power of 2 nor a multiple of the block_size defined below
vector_size = 2**24 + 11

device = cp.cuda.Device()
## Create device arrays of uniform random float32 values as input, and an array of zeros
## as the result vector
a = cp.random.uniform(-1, 1, vector_size)
b = cp.random.uniform(-1, 1, vector_size)
c = cp.zeros_like(a)

block_size = 256
grid_size = int(np.ceil(vector_size/block_size))
vecadd[grid_size, block_size](a, b, c)

## synchronize the CPU thread and the GPU to ensure that the kernel has completed
## this is included to illustrate good practices, even though the copy below would implicitly wait for
## the kernel to complete
device.synchronize()

## Copy all 3 arrays to the CPU as ndarrays
a_np = cp.asnumpy(a)
b_np = cp.asnumpy(b)
c_np = cp.asnumpy(c)

## Perform the copy on the CPU to verify the answer
expected = a_np + b_np

## Test that the answer is correct, within floating point epsilon
np.testing.assert_array_almost_equal(c_np, expected)

## The assert will print diagnostics and abort
## so this only prints if the assertion passes
print("Test succeeded")
```

在这个示例中，输入数组 `A` 和 `B` 由 CuPy 在 GPU 上创建并初始化为随机值。它们只在代码结尾被复制到 CPU，这样 CPU 也可以执行向量加法，并验证 CPU 与 GPU 的答案是否一致。

## 2.2.6 CUDA Python 中的错误检查（Error Checking in CUDA Python）

从内存分配、内存复制到 kernel launch，任何影响 GPU 的操作都有可能产生错误条件。正如 C++ 的[第 2.1.7 节](./intro-to-cuda-cpp.html)所示，在与 GPU 交互的过程中确认没有发生错误是最佳实践。

在 Python 中，CUDA 错误会抛出异常；如果异常没有捕获，就会终止程序。可以使用普通 Python 语法捕获异常。下面的例子与上面的向量加法相同，但故意加入了一个错误：每个 block 的线程数设为 2048，大于当前任何 GPU 能运行的数量。这会使 kernel launch 失败并抛出异常，代码会捕获该异常。

```python
import numpy as np
from numba import cuda
import cupy as cp


## Defines a CUDA kernel to perform C = A + B vector addition
@cuda.jit
def vecadd(A, B, C):
    work_index = cuda.grid(1)
    C[work_index] = A[work_index] + B[work_index]


try:
    vector_size = 2**24 + 11

    device = cp.cuda.Device()
    a = cp.random.uniform(-1, 1, vector_size)
    b = cp.random.uniform(-1, 1, vector_size)
    c = cp.zeros_like(a)

    ## this block size is too large for any current GPUs
    block_size = 2048
    grid_size = int(np.ceil(vector_size/block_size))
    # Error: launching kernel with invalid block size
    vecadd[grid_size, block_size](a, b, c)

    device.synchronize()
    print("Test did not encounter any errors")

except Exception as e:
    print(f"Exception occurred: {e}")
```

运行上面的代码会捕获并显示错误：

```bash
$ python3 vecadd_error.py
Exception occurred: CUDA_ERROR_INVALID_VALUE: This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.
```

程序捕获异常后正常退出。如果去掉 `try:` 和 `except:`，程序会异常退出，并把 traceback 转储到控制台；traceback 应该会显示同一个错误。

## 本节导航

- [2.1 CUDA C++ 入门](./intro-to-cuda-cpp.html)
- [2.3 编写 SIMT Kernel](./writing-simt-kernels.html)
- [2.4 编写 Tile Kernel](./writing-tile-kernels.html)
- [2.5 异步执行](./asynchronous-execution.html)
