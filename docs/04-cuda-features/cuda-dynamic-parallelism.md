---
title: 4.18 CUDA 动态并行（CUDA Dynamic Parallelism）
description: CUDA 动态并行的执行环境、内存一致性、编程接口和 PTX 设备端启动。
---

# 4.18 CUDA 动态并行（CUDA Dynamic Parallelism）

## 4.18.1 简介（Introduction）

### 4.18.1.1 概览（Overview）

CUDA 动态并行（通常缩写为 CDP）是 CUDA 编程模型的一项功能，允许 GPU 上运行的代码创建新的 GPU 工作。也就是说，已经在 GPU 上运行的设备代码可以通过额外的 kernel 启动向 GPU 添加新工作。由于启动配置决策可以由设备上执行的线程在运行时作出，这项功能可以减少在主机与设备之间传递执行控制权和数据的需要。

Kernel 可以在运行时生成依赖数据的并行工作。在 CDP 加入 CUDA 之前，某些算法和编程模式需要修改，以消除递归、不规则循环结构，或其他不适合扁平单层并行性的结构。使用 CUDA 动态并行可以更自然地表达这些程序结构。

::: note

本节介绍 CUDA 动态并行的更新版本，有时称为 CDP2；CDP2 是 CUDA 12.0 及更高版本中的默认版本。对于计算能力（CC）9.0 及更高版本的设备，CDP2 是唯一可用的 CUDA 动态并行版本。对于 CC 低于 9.0 的设备，开发者仍可以通过编译器参数 `-DCUDA_FORCE_CDP1_IF_SUPPORTED` 选择传统 CUDA 动态并行 CDP1。CDP1 文档见 [CUDA Programming Guide 的旧版本](https://developer.nvidia.com/cuda-toolkit-archive)。CDP1 计划在未来版本的 CUDA 中移除。

:::

## 4.18.2 执行环境（Execution Environment）

CUDA 中的动态并行允许 GPU 线程配置、启动新的 grid，并隐式地与新的 grid 同步。grid 是一次 kernel 启动的实例，其中包含线程块的具体形状以及线程块组成的 grid。请注意，kernel 函数本身与该 kernel 的一次具体调用（即一个 grid）是不同的，后续章节会用到这一点。

### 4.18.2.1 父 Grid 与子 Grid（Parent and Child Grids）

配置并启动新 grid 的设备线程属于父 grid。该次调用创建的新 grid 称为子 grid。

子 grid 的调用和完成是正确嵌套的：只有由父 grid 中线程创建的所有子 grid 都完成后，父 grid 才会被视为完成；runtime 保证父 grid 与子 grid 之间存在隐式同步。

![父子启动嵌套](/images/chapter-04/parent-child-launch-nesting.png)

图 57：父子启动嵌套（Parent-Child Launch Nesting）

### 4.18.2.2 CUDA 原语的作用域（Scope of CUDA Primitives）

CUDA 动态并行依赖 [CUDA Device Runtime](../05-technical-appendices/device-callable-apis.html#cuda-device-runtime)，它允许在设备代码中调用一组有限的 API。这些 API 在语法上类似 CUDA Runtime API，但可以在设备代码中使用。设备 runtime API 的行为类似其主机端对应 API，但二者存在一些差异。这些差异记录在 [API 参考](../05-technical-appendices/device-callable-apis.html#device-runtime-api-reference) 一节的表中。

在主机和设备两端，CUDA runtime 都提供启动 kernel 的 API，以及通过流和事件跟踪启动之间依赖关系的 API。在设备端，已启动的 kernel 和 CUDA 对象对调用 grid 中的所有线程可见。例如，一个线程可以创建流，同一 grid 中的任意其他线程都可以使用该流。但是，通过设备 API 调用创建的流和事件等 CUDA 对象，只在创建它们的 grid 内有效。

### 4.18.2.3 流与事件（Streams and Events）

CUDA *流（Streams）* 和 *事件（Events）* 可以控制 kernel 启动之间的依赖关系：在同一流中启动的 kernel 按顺序执行，事件可以用来在不同流之间创建依赖关系。设备上创建的流和事件具有完全相同的用途。

在某个 grid 内创建的流和事件存在于该 grid 的作用域中；在创建它们的 grid 之外使用时，行为未定义。如上所述，当 grid 退出时，该 grid 启动的所有工作都会隐式同步；向流中启动的工作也包含在其中，并且所有依赖关系都会得到适当解析。在 grid 作用域之外修改过的流上执行操作，其行为未定义。

在主机上创建的流和事件，在任意 kernel 中使用时行为未定义；同样，父 grid 创建的流和事件在子 grid 中使用时行为也未定义。

### 4.18.2.4 顺序与并发（Ordering and Concurrency）

设备 runtime 的 kernel 启动顺序遵循 CUDA 流排序语义。在一个 grid 内，向同一流中启动的所有 kernel（[Fire-and-Forget 流](../05-technical-appendices/device-callable-apis.html#fire-and-forget-stream)除外）都按顺序执行。当同一 grid 中的多个线程向同一流启动 kernel 时，流内顺序取决于 grid 中的线程调度；可以使用 `__syncthreads()` 等同步原语控制这种调度。

请注意，命名流由一个 grid 内的所有线程共享，但隐式 *NULL* 流只由一个线程块内的所有线程共享。如果一个线程块中的多个线程向隐式流启动 kernel，这些启动会按顺序执行。如果不同线程块中的线程向隐式流启动 kernel，这些启动可能并发执行。如果希望一个线程块内多个线程的启动具有并发性，应使用显式命名流。

设备 runtime 不会向 CUDA 执行模型引入新的并发保证。也就是说，设备不保证任意数量的不同线程块之间并发执行。

这种没有并发保证的情况同样适用于父 grid 与其子 grid。父 grid 启动子 grid 后，只要流依赖满足且硬件资源可用，子 grid 就可能开始执行；但在父 grid 到达隐式同步点之前，并不保证子 grid 会开始执行。

并发性可能随设备配置、应用工作负载和 runtime 调度而变化。因此，依赖不同线程块之间存在任何并发性都是不安全的。

## 4.18.3 内存一致性与一致性保证（Memory Coherence and Consistency）

父 grid 与子 grid 共享同一份全局内存和常量内存存储，但拥有各自独立的局部内存和共享内存。下表展示了哪些内存空间允许父 grid 和子 grid 使用相同指针访问。子 grid 永远不能访问父 grid 的局部内存或共享内存，父 grid 也不能访问子 grid 的局部内存或共享内存。

| 内存空间 | 父/子 grid 是否使用相同指针？ |
| --- | --- |
| 全局内存（Global Memory） | 是 |
| 映射内存（Mapped Memory） | 是 |
| 局部内存（Local Memory） | 否 |
| 共享内存（Shared Memory） | 否 |
| 纹理内存（Texture Memory） | 是（只读） |

表 26：动态并行：父 grid 与子 grid 之间的内存作用域可访问性（Dynamic Parallelism: Memory Scope Accessibility Between Parent and Child Grids）

### 4.18.3.1 全局内存（Global Memory）

父 grid 和子 grid 对全局内存具有一致的访问，但父与子之间的一致性保证较弱。在子 grid 的执行过程中，只有一个时间点能保证它观察到的内存视图与父线程完全一致：父 grid 调用子 grid 的时刻。

子 grid 被调用前，父线程执行的所有全局内存操作对子 grid 可见。移除 `cudaDeviceSynchronize()` 后，父 grid 无法再访问子 grid 中线程所作的修改。父 grid 退出前，要访问子 grid 中线程所作的修改，唯一方法是向 `cudaStreamTailLaunch` 流启动一个 kernel。

下面的示例中，执行 `child_launch` 的子 grid 只能保证看到子 grid 启动前对 `data` 作出的修改。由于父 grid 的线程 0 执行启动操作，子 grid 将与父 grid 的线程 0 所看到的内存保持一致。由于第一个 `__syncthreads()` 调用，子 grid 将看到 `data[0]=0`、`data[1]=1`、…、`data[255]=255`（如果没有 `__syncthreads()` 调用，则只能保证子 grid 看到 `data[0]=0`）。子 grid 只能保证在隐式同步点返回。这意味着，子 grid 中线程所作的修改永远不能保证对父 grid 可用。要访问 `child_launch` 所作的修改，需要向 `cudaStreamTailLaunch` 流启动 `tail_launch` kernel。

```cpp
__global__ void tail_launch(int *data) {
   data[threadIdx.x] = data[threadIdx.x]+1;
}

__global__ void child_launch(int *data) {
   data[threadIdx.x] = data[threadIdx.x]+1;
}

__global__ void parent_launch(int *data) {
   data[threadIdx.x] = threadIdx.x;

   __syncthreads();

   if (threadIdx.x == 0) {
       child_launch<<< 1, 256 >>>(data);
       tail_launch<<< 1, 256, 0, cudaStreamTailLaunch >>>(data);
   }
}

void host_launch(int *data) {
    parent_launch<<< 1, 256 >>>(data);
}
```

### 4.18.3.2 映射内存（Mapped Memory）

映射系统内存具有与全局内存相同的一致性和一致性保证，并遵循上文所述语义。Kernel 不能分配或释放映射内存，但可以使用由主机程序传入的映射内存指针。

### 4.18.3.3 共享内存和局部内存（Shared and Local Memory）

共享内存和局部内存分别属于线程块和线程私有，在父 grid 与子 grid 之间不可见且不保持一致。在所属作用域之外引用这两类内存中的对象，其行为未定义，并可能导致错误。

如果 NVIDIA 编译器能够检测出传递给 kernel 启动参数的是指向局部内存或共享内存的指针，它会尝试发出警告。在运行时，程序员可以使用 `__isGlobal()` 内建函数判断指针是否引用全局内存，从而确定该指针是否可以安全地传递给子 grid 启动。

`cudaMemcpy*Async()` 或 `cudaMemset*Async()` 调用可能会在设备上启动新的子 kernel，以保持流语义。因此，将共享内存或局部内存指针传递给这些 API 是非法的，并会返回错误。

### 4.18.3.4 局部内存（Local Memory）

局部内存是执行中线程的私有存储，对该线程之外不可见。启动子 kernel 时，将指向局部内存的指针作为启动参数传递是非法的。子 grid 解引用此类局部内存指针的结果未定义。

例如，下面的代码是非法的；如果 `child_launch` 访问 `x_array`，行为未定义：

```cpp
int x_array[10];       // 在父线程的局部内存中创建 x_array
child_launch<<< 1, 1 >>>(x_array);
```

程序员有时很难知道某个变量何时会被编译器放入局部内存。一般来说，传递给子 kernel 的所有存储都应显式从全局内存堆中分配，例如使用 `cudaMalloc()`、`new()`，或者在全局作用域声明 `__device__` 存储。例如：

```cpp
// 正确：“value”是全局存储
__device__ int value;
__device__ void x() {
    value = 5;
    child<<< 1, 1 >>>(&value);
}
```

```cpp
// 无效：“value”是局部存储
__device__ void y() {
    int value = 5;
    child<<< 1, 1 >>>(&value);
}
```

#### 4.18.3.4.1 纹理内存（Texture Memory）

对纹理所映射的全局内存区域执行写入时，该写入与纹理访问之间不保持一致。纹理内存的一致性会在调用子 grid 以及子 grid 完成时建立。这意味着，子 kernel 启动前对内存的写入会反映在子 grid 的纹理内存访问中。与上面的全局内存类似，子 grid 对内存的写入永远不能保证反映在父 grid 的纹理内存访问中。父 grid 退出前，要访问子 grid 中线程所作的修改，唯一方法是向 `cudaStreamTailLaunch` 流启动一个 kernel。父 grid 与子 grid 的并发访问可能导致数据不一致。

## 4.18.4 编程接口（Programming Interface）

### 4.18.4.1 基础（Basics）

下面的示例展示了一个包含动态并行的简单 *Hello World* 程序：

```cpp
#include <stdio.h>

__global__ void childKernel()
{
    printf("Hello ");
}

__global__ void tailKernel()
{
    printf("World!\n");
}

__global__ void parentKernel()
{
    // 启动子 kernel
    childKernel<<<1,1>>>();
    if (cudaSuccess != cudaGetLastError()) {
        return;
    }

    // 将 tail kernel 启动到 cudaStreamTailLaunch 流
    // 隐式同步：等待子 kernel 完成
    tailKernel<<<1,1,0,cudaStreamTailLaunch>>>();

}

int main(int argc, char *argv[])
{
    // 启动父 kernel
    parentKernel<<<1,1>>>();
    if (cudaSuccess != cudaGetLastError()) {
        return 1;
    }

    // 等待父 kernel 完成
    if (cudaSuccess != cudaDeviceSynchronize()) {
        return 2;
    }

    return 0;
}
```

可以在命令行中用一步完成该程序的构建：

```text
$ nvcc -arch=sm_75 -rdc=true hello_world.cu -o hello -lcudadevrt
```

### 4.18.4.2 CDP 的 C++ 语言接口（C++ Language Interface for CDP）

使用 CUDA C++ 实现动态并行的 CUDA kernel 可用的语言接口和 API 称为 [CUDA Device Runtime](../05-technical-appendices/device-callable-apis.html#cuda-device-runtime)。

在可能的情况下，该接口保留 CUDA Runtime API 的语法和语义，以便复用可以在主机或设备环境中运行的例程代码。

与 CUDA C++ 中的所有代码一样，本节介绍的 API 和代码都是逐线程代码。这使每个线程都能独立、动态地决定下一步执行哪个 kernel 或操作。在一个线程块内的线程之间，不需要同步即可执行这里提供的任何设备 runtime API；因此，设备 runtime API 函数可以在任意分歧的 kernel 代码中调用而不会死锁。

#### 4.18.4.2.1 设备端 Kernel 启动（Device-Side Kernel Launch）

可以使用标准 CUDA `<<< >>>` 语法从设备端启动 kernel：

```cpp
kernel_name<<< Dg, Db, Ns, S >>>([kernel arguments]);
```

- `Dg` 类型为 `dim3`，指定 grid 的维度和大小。
- `Db` 类型为 `dim3`，指定每个线程块的维度和大小。
- `Ns` 类型为 `size_t`，指定本次调用中每个线程块动态分配的共享内存字节数；这部分内存是在静态分配内存之外额外分配的。`Ns` 是可选参数，默认值为 0。
- `S` 类型为 `cudaStream_t`，指定与本次调用关联的流。该流必须在执行调用的同一个 grid 中分配。`S` 是可选参数，默认使用 NULL 流。

##### 4.18.4.2.1.1 启动是异步的（Launches are Asynchronous）

与主机端启动相同，所有设备端 kernel 启动相对于启动线程都是异步的。也就是说，`<<<>>>` 启动命令会立即返回，启动线程会继续执行，直到到达隐式启动同步点，例如启动到 `cudaStreamTailLaunch` 流中的 kernel（[Tail Launch 流](../05-technical-appendices/device-callable-apis.html#tail-launch-stream)）。子 grid 可以在启动后的任意时间开始执行，但在启动线程到达隐式启动同步点之前，并不保证它会开始执行。

与主机端启动类似，向不同流中启动的工作可能并发运行，但实际并发不受保证。依赖子 kernel 之间并发的程序不受 CUDA 编程模型支持，其行为未定义。

##### 4.18.4.2.1.2 启动环境配置（Launch Environment Configuration）

所有全局设备配置设置（例如 `cudaDeviceGetCacheConfig()` 返回的共享内存和 L1 缓存大小，以及 `cudaDeviceGetLimit()` 返回的设备限制）都会从父环境继承。同样，栈大小等设备限制会保持已配置的值。

对于由主机启动的 kernel，在主机端设置的每 kernel 配置优先于全局设置。从设备端启动该 kernel 时也会使用这些配置。不可能从设备端重新配置 kernel 的环境。

#### 4.18.4.2.2 事件（Events）

CUDA 事件只支持流间同步能力。这意味着支持 `cudaStreamWaitEvent()`，但不支持 `cudaEventSynchronize()`、`cudaEventElapsedTime()` 和 `cudaEventQuery()`。由于不支持 `cudaEventElapsedTime()`，必须通过 `cudaEventCreateWithFlags()` 创建 cudaEvent，并传入 `cudaEventDisableTiming` 标志。

与流一样，事件对象可以由创建它们的 grid 内所有线程共享，但它们属于该 grid，不能传递给其他 kernel。事件句柄不能保证在不同 grid 之间唯一，因此在没有创建该事件的 grid 中使用事件句柄会导致未定义行为。

#### 4.18.4.2.3 同步（Synchronization）

如果调用线程需要与其他线程调用的子 grid 同步，则程序必须执行足够的线程间同步，例如通过 CUDA 事件进行同步。

由于无法从父线程显式同步子工作，因此无法保证子 grid 中发生的修改对父 grid 内线程可见。

#### 4.18.4.2.4 设备管理（Device Management）

只有 kernel 正在运行的设备可以从该 kernel 中控制。因此，`cudaSetDevice()` 等设备 API 不受设备 runtime 支持。从 GPU 端看到的活动设备（由 `cudaGetDevice()` 返回）与主机系统看到的设备编号相同。`cudaDeviceGetAttribute()` 调用可以请求另一个设备的信息，因为该 API 允许在调用参数中指定设备 ID。请注意，设备 runtime 不提供用于一次获取全部属性的 `cudaGetDeviceProperties()` API；必须逐项查询属性。

## 4.18.5 编程指南（Programming Guidelines）

### 4.18.5.1 性能（Performance）

#### 4.18.5.1.1 启用动态并行的 Kernel 开销（Dynamic-parallelism-enabled Kernel Overhead）

控制动态启动时处于活动状态的系统软件，可能会对当时正在运行的任意 kernel 产生开销，无论该 kernel 是否自行启动其他 kernel。这种开销来自设备 runtime 的执行跟踪和管理软件，可能导致性能下降。通常，链接设备 runtime 库的应用都会产生这种开销。

### 4.18.5.2 实现限制（Implementation Restrictions and Limitations）

动态并行保证本文档描述的所有语义，但某些硬件和软件资源取决于具体实现，会限制使用设备 runtime 的程序的规模、性能和其他属性。

#### 4.18.5.2.1 运行时（Runtime）

##### 4.18.5.2.1.1 内存占用（Memory Footprint）

设备 runtime 系统软件会为各种管理用途预留内存，尤其会为跟踪待处理的 grid 启动预留内存。可以通过配置控制减小该预留区域，但代价是受到某些启动限制。具体信息请参见下面的[“配置选项”](../05-technical-appendices/device-callable-apis.html#device-runtime-configuration-options)。

##### 4.18.5.2.1.2 待处理的 Kernel 启动（Pending Kernel Launches）

启动 kernel 时，所有相关配置和参数数据都会被跟踪，直到 kernel 完成。这些数据存储在系统管理的启动池中。

可以在主机端调用 `cudaDeviceSetLimit()` 并指定 `cudaLimitDevRuntimePendingLaunchCount`，配置固定大小启动池的大小。

### 4.18.5.3 兼容性与互操作性（Compatibility and Interoperability）

CDP2 是默认版本。对于计算能力低于 9.0 的设备，可以使用 `-DCUDA_FORCE_CDP1_IF_SUPPORTED` 编译函数，从而选择不使用 CDP2。

|  | 使用 CUDA 12.0 及更高版本编译的函数（默认） | 使用 CUDA 12.0 之前版本编译的函数，或使用 CUDA 12.0 及更高版本且指定 `-DCUDA_FORCE_CDP1_IF_SUPPORTED` 编译的函数 |
| --- | --- | --- |
| 编译 | 如果设备代码引用 `cudaDeviceSynchronize`，则编译错误。 | 如果代码引用 `cudaStreamTailLaunch` 或 `cudaStreamFireAndForget`，则编译错误。如果设备代码引用 `cudaDeviceSynchronize` 且代码针对 sm_90 或更高版本编译，则编译错误。 |
| 计算能力 < 9.0 | 使用新接口。 | 使用传统接口。 |
| 计算能力 9.0 及更高版本 | 使用新接口。 | 使用新接口。如果函数在设备代码中引用 `cudaDeviceSynchronize`，则函数加载返回 `cudaErrorSymbolNotFound`（例如，代码针对计算能力低于 9.0 的设备编译，却通过 JIT 在计算能力 9.0 或更高的设备上运行时，可能发生这种情况）。 |

使用 CDP1 和 CDP2 的函数可以在同一上下文中同时加载和运行。CDP1 函数可以使用 CDP1 专用功能（例如 `cudaDeviceSynchronize`），CDP2 函数可以使用 CDP2 专用功能（例如 tail launch 和 fire-and-forget launch）。

使用 CDP1 的函数不能启动使用 CDP2 的函数，反之亦然。如果一个本应使用 CDP1 的函数在调用图中包含一个使用 CDP2 的函数，或者反过来，那么函数加载期间会产生 `cudaErrorCdpVersionMismatch`。

本文档不包含传统 CDP1 的行为。有关 CDP1 的信息，请参阅 [CUDA Programming Guide 的旧版本](https://developer.nvidia.com/cuda-toolkit-archive)。

## 4.18.6 从 PTX 进行设备端启动（Device-side Launch from PTX）

前面各节介绍了如何使用 [CUDA Device Runtime](../05-technical-appendices/device-callable-apis.html#cuda-device-runtime) 实现动态并行。也可以从 PTX 执行动态并行。对于面向 *Parallel Thread Execution*（PTX）并计划在其语言中支持 *Dynamic Parallelism* 的编程语言和编译器实现者，本节提供了在 PTX 层支持 kernel 启动所需的底层细节。

### 4.18.6.1 Kernel 启动 API（Kernel Launch APIs）

可以使用两个可从 PTX 访问的 API 实现设备端 kernel 启动：`cudaLaunchDevice()` 和 `cudaGetParameterBuffer()`。`cudaLaunchDevice()` 使用参数缓冲区启动指定 kernel；该参数缓冲区通过调用 `cudaGetParameterBuffer()` 获取，并填入传递给所启动 kernel 的参数。如果所启动的 kernel 不接受任何参数，参数缓冲区可以为 NULL，即无需调用 `cudaGetParameterBuffer()`。

#### 4.18.6.1.1 cudaLaunchDevice 启动 API（cudaLaunchDevice）

在使用 `cudaLaunchDevice()` 之前，必须在 PTX 层以如下两种形式之一声明它。下面展示的是 `.address_size` 为 64 时的声明：

```ptx
// .address_size 为 64 时 cudaLaunchDevice() 的 PTX 层声明
.extern .func(.param .b32 func_retval0) cudaLaunchDevice
(
  .param .b64 func,
  .param .b64 parameterBuffer,
  .param .align 4 .b8 gridDimension[12],
  .param .align 4 .b8 blockDimension[12],
  .param .b32 sharedMemSize,
  .param .b64 stream
)
;
```

下面的 CUDA 层声明会映射到上述两种 PTX 层声明之一，并位于系统头文件 `cuda_device_runtime_api.h` 中。该函数定义在系统库 `cudadevrt` 中；要使用设备端 kernel 启动功能，必须将该库链接到程序中。

```cpp
// cudaLaunchDevice() 的 CUDA 层声明
extern "C" __device__
cudaError_t cudaLaunchDevice(void *func, void *parameterBuffer,
                             dim3 gridDimension, dim3 blockDimension,
                             unsigned int sharedMemSize,
                             cudaStream_t stream);
```

第一个参数是要启动的 kernel 指针，第二个参数是保存所启动 kernel 实际参数的参数缓冲区。参数缓冲区的布局在下面的[“参数缓冲区布局”](#41862-参数缓冲区布局)中说明。其他参数指定启动配置，包括 grid 维度、线程块维度、共享内存大小和与启动关联的流；有关启动配置的详细描述，请参阅[“Kernel 配置”](../05-technical-appendices/cpp-language-extensions.html#execution-configuration)。

#### 4.18.6.1.2 cudaGetParameterBuffer 参数缓冲区 API（cudaGetParameterBuffer）

在使用 `cudaGetParameterBuffer()` 之前，必须在 PTX 层声明它。根据地址大小，PTX 层声明必须采用以下两种形式之一：

```ptx
// .address_size 为 64 时 cudaGetParameterBuffer() 的 PTX 层声明
.extern .func(.param .b64 func_retval0) cudaGetParameterBuffer
(
  .param .b64 alignment,
  .param .b64 size
)
;
```

下面的 CUDA 层 `cudaGetParameterBuffer()` 声明会映射到上述 PTX 层声明：

```cpp
// cudaGetParameterBuffer() 的 CUDA 层声明
extern "C" __device__
void *cudaGetParameterBuffer(size_t alignment, size_t size);
```

第一个参数指定参数缓冲区的对齐要求，第二个参数指定以字节为单位的大小要求。在当前实现中，`cudaGetParameterBuffer()` 返回的参数缓冲区始终保证按 64 字节对齐，并且会忽略对齐要求参数。不过，为保证未来的可移植性，建议将正确的对齐要求值传递给 `cudaGetParameterBuffer()`；该值应是放入参数缓冲区的所有参数中最大的对齐要求。

### 4.18.6.2 参数缓冲区布局（Parameter Buffer Layout）

禁止对参数缓冲区中的参数重新排序，并且放入参数缓冲区的每个独立参数都必须对齐。也就是说，每个参数都必须放置在参数缓冲区的第 *n* 个字节处，其中 *n* 是大于前一个参数占用的最后一个字节偏移量的、参数大小的最小倍数。参数缓冲区的最大大小为 4 KB。

有关 CUDA 编译器生成的 PTX 代码的更详细说明，请参阅 PTX 3.5 规范。
