---
title: 3.3 CUDA 驱动 API
description: CUDA Driver API 的上下文、模块、Kernel 执行，以及与 Runtime API 的互操作
---

# 3.3 CUDA 驱动 API（The CUDA Driver API）

本指南前面的章节介绍了 CUDA Runtime。正如[CUDA Runtime API 与 CUDA Driver API](../01-introduction/cuda-platform.html#cuda-platform-driver-and-runtime)中所述，CUDA Runtime 构建在更底层的 CUDA Driver API 之上。本节介绍 CUDA Runtime API 与 Driver API 之间的一些差异，以及如何混合使用它们。大多数应用无需直接接触 CUDA Driver API，也能获得完整性能。不过，Driver API 有时会比 Runtime API 更早提供新接口；而一些高级接口，例如[虚拟内存管理](../04-cuda-features/virtual-memory-management.html#virtual-memory-management)，也只在 Driver API 中公开。

Driver API 实现在 `cuda` 动态库（`cuda.dll` 或 `cuda.so`）中。安装设备驱动程序时，该库会被复制到系统中。Driver API 的所有入口点都以 `cu` 开头。

这是一个基于句柄的命令式 API：大多数对象都由不透明句柄引用，应用程序可以将这些句柄传递给函数来操作相应对象。

Driver API 中可用的对象总结如下。

| 对象 | 句柄 | 描述 |
| --- | --- | --- |
| Device（设备） | `CUdevice` | 支持 CUDA 的设备 |
| Context（上下文） | `CUcontext` | 大致等价于 CPU 进程 |
| Module（模块） | `CUmodule` | 大致等价于动态库 |
| Function（函数） | `CUfunction` | Kernel |
| Heap memory（堆内存） | `CUdeviceptr` | 指向设备内存的指针 |
| CUDA array（CUDA 数组） | `CUarray` | 设备上一维或二维数据的不透明容器，可通过 texture 或 surface 引用读取 |
| Texture object（Texture 对象） | `CUtexref` | 描述如何解释 texture memory 数据的对象 |
| Surface reference（Surface 引用） | `CUsurfref` | 描述如何读写 CUDA 数组的对象 |
| Stream（Stream） | `CUstream` | 描述 CUDA stream 的对象 |
| Event（Event） | `CUevent` | 描述 CUDA event 的对象 |

表 6：CUDA Driver API 中可用的对象（Objects Available in the CUDA Driver API）。

调用 Driver API 的任何函数之前，都必须先通过 `cuInit()` 初始化 Driver API。随后必须创建一个附加到特定设备的 CUDA context，并将其设为调用线程的当前 context；具体过程见[3.3.1 Context](#331-context)。

在 CUDA context 中，Kernel 会由主机代码显式加载为 PTX 或二进制对象，如[3.3.2 Module](#332-module)所述。因此，用 C++ 编写的 Kernel 必须单独编译为 *PTX* 或二进制对象。Kernel 使用[3.3.3 Kernel Execution](#333-kernel-execution)所述的 API 入口点启动。

如果应用程序要运行在未来的设备架构上，就必须加载 *PTX*，而不是二进制代码。这是因为二进制代码与具体架构绑定，因而与未来架构不兼容；而 *PTX* 代码会在加载时由设备驱动程序编译为二进制代码。

下面是[Kernel](../02-programming-gpus/intro-to-cuda-cpp.html#kernels)示例中使用 Driver API 编写的主机代码：

```cpp
int main()
{
    int N = ...;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    // Initialize input vectors
    ...

    // Initialize
    cuInit(0);

    // Get number of devices supporting CUDA
    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        exit (0);
    }

    // Get handle for device 0
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, 0);

    // Create context
    CUcontext cuContext;
    cuCtxCreate(&cuContext, 0, cuDevice);

    // Create module from binary file
    CUmodule cuModule;
    cuModuleLoad(&cuModule, "VecAdd.ptx");

    // Allocate vectors in device memory
    CUdeviceptr d_A;
    cuMemAlloc(&d_A, size);
    CUdeviceptr d_B;
    cuMemAlloc(&d_B, size);
    CUdeviceptr d_C;
    cuMemAlloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cuMemcpyHtoD(d_A, h_A, size);
    cuMemcpyHtoD(d_B, h_B, size);

    // Get function handle from module
    CUfunction vecAdd;
    cuModuleGetFunction(&vecAdd, cuModule, "VecAdd");

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    void* args[] = { &d_A, &d_B, &d_C, &N };
    cuLaunchKernel(vecAdd,
                   blocksPerGrid, 1, 1, threadsPerBlock, 1, 1,
                   0, 0, args, 0);

    ...
}
```

完整代码见 CUDA 示例 `vectorAddDrv`。

## 3.3.1 上下文（Context）

CUDA context 类似于 CPU 进程。Driver API 执行的所有资源和操作都封装在 CUDA context 中；context 销毁时，系统会自动清理这些资源。除了 module、texture 引用或 surface 引用等对象以外，每个 context 还有自己独立的地址空间。因此，不同 context 中的 `CUdeviceptr` 值指向不同的内存位置。

一个主机线程在同一时间只能有一个当前设备 context。通过 `cuCtxCreate()` 创建 context 时，该 context 会被设为调用线程的当前 context。对 context 执行操作的 CUDA 函数（大多数不涉及设备枚举或 context 管理的函数）如果在线程中没有有效的当前 context，就会返回 `CUDA_ERROR_INVALID_CONTEXT`。

每个主机线程都有一个当前 context 栈。`cuCtxCreate()` 会把新 context 压入栈顶。可以调用 `cuCtxPopCurrent()` 将 context 从主机线程分离。此时该 context 处于“floating”（浮动）状态，可以作为任意主机线程的当前 context 被重新压入。`cuCtxPopCurrent()` 还会恢复之前的当前 context（如果存在）。

每个 context 还维护一个使用计数。`cuCtxCreate()` 创建使用计数为 1 的 context；`cuCtxAttach()` 会增加使用计数，`cuCtxDetach()` 会减少使用计数。当调用 `cuCtxDetach()` 或 `cuCtxDestroy()` 后使用计数变为 0 时，context 会被销毁。

Driver API 与 Runtime API 互操作。Runtime 管理的 primary context（见 [Runtime Initialization](../02-programming-gpus/intro-to-cuda-cpp.html#intro-cpp-runtime-initialization)）可以通过 Driver API 的 `cuDevicePrimaryCtxRetain()` 访问。

使用计数便于多个第三方编写的代码在同一个 context 中互操作。例如，如果加载了三个要使用同一 context 的库，每个库都会调用 `cuCtxAttach()` 增加使用计数，并在使用完 context 后调用 `cuCtxDetach()` 减少使用计数。对于大多数库，通常假定应用程序会在加载或初始化库之前创建 context；这样，应用程序可以根据自己的策略创建 context，而库只需操作传递给它的 context。希望创建自己的 context、且不让 API 客户端知道这一点的库（这些客户端可能已经创建了自己的 context，也可能没有）应使用 `cuCtxPushCurrent()` 和 `cuCtxPopCurrent()`，如下图所示。

![图 23：库上下文管理（Library Context Management）](/images/chapter-03/library-context-management.png)

图 23：库上下文管理（Library Context Management）。应用程序可以创建并维护自己的 context；需要独立 context 的库则通过 push/pop 操作暂时切换当前 context，完成工作后恢复调用线程原来的 context。

## 3.3.2 模块（Module）

Module 是可动态加载的设备代码和数据包，类似于 Windows 中的 DLL，由 `nvcc` 输出（见 [使用 NVCC 编译](../02-programming-gpus/intro-to-cuda-cpp.html#compilation-with-nvcc)）。所有符号（包括函数、全局变量以及 texture 或 surface 引用）的名称都保存在 module 作用域中，因此由独立第三方编写的 module 可以在同一个 CUDA context 中互操作。

下面的代码加载一个 module，并获取某个 Kernel 的句柄：

```cpp
CUmodule cuModule;
cuModuleLoad(&cuModule, "myModule.ptx");
CUfunction myKernel;
cuModuleGetFunction(&myKernel, cuModule, "MyKernel");
```

下面的代码从 PTX 代码编译并加载一个新 module，同时解析编译错误：

```cpp
#define BUFFER_SIZE 8192
CUmodule cuModule;
CUjit_option options[3];
void* values[3];
char* PTXCode = "some PTX code";
char error_log[BUFFER_SIZE];
int err;
options[0] = CU_JIT_ERROR_LOG_BUFFER;
values[0]  = (void*)error_log;
options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
values[1]  = (void*)BUFFER_SIZE;
options[2] = CU_JIT_TARGET_FROM_CUCONTEXT;
values[2]  = 0;
err = cuModuleLoadDataEx(&cuModule, PTXCode, 3, options, values);
if (err != CUDA_SUCCESS)
    printf("Link error:\n%s\n", error_log);
```

下面的代码从多个 PTX 代码编译、链接并加载一个新 module，同时解析链接错误和编译错误：

```cpp
#define BUFFER_SIZE 8192
CUmodule cuModule;
CUjit_option options[6];
void* values[6];
float walltime;
char error_log[BUFFER_SIZE], info_log[BUFFER_SIZE];
char* PTXCode0 = "some PTX code";
char* PTXCode1 = "some other PTX code";
CUlinkState linkState;
int err;
void* cubin;
size_t cubinSize;
options[0] = CU_JIT_WALL_TIME;
values[0] = (void*)&walltime;
options[1] = CU_JIT_INFO_LOG_BUFFER;
values[1] = (void*)info_log;
options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
values[2] = (void*)BUFFER_SIZE;
options[3] = CU_JIT_ERROR_LOG_BUFFER;
values[3] = (void*)error_log;
options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
values[4] = (void*)BUFFER_SIZE;
options[5] = CU_JIT_LOG_VERBOSE;
values[5] = (void*)1;
cuLinkCreate(6, options, values, &linkState);
err = cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
                    (void*)PTXCode0, strlen(PTXCode0) + 1, 0, 0, 0, 0);
if (err != CUDA_SUCCESS)
    printf("Link error:\n%s\n", error_log);
err = cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
                    (void*)PTXCode1, strlen(PTXCode1) + 1, 0, 0, 0, 0);
if (err != CUDA_SUCCESS)
    printf("Link error:\n%s\n", error_log);
cuLinkComplete(linkState, &cubin, &cubinSize);
printf("Link completed in %fms. Linker Output:\n%s\n", walltime, info_log);
cuModuleLoadData(cuModule, cubin);
cuLinkDestroy(linkState);
```

使用多个线程可以加速 module 链接/加载过程中的一部分操作，包括加载 cubin。下面的代码使用 `CU_JIT_BINARY_LOADER_THREAD_COUNT` 加速 module 加载：

```cpp
#define BUFFER_SIZE 8192
CUmodule cuModule;
CUjit_option options[3];
void* values[3];
char* cubinCode = "some cubin code";
char error_log[BUFFER_SIZE];
int err;
options[0] = CU_JIT_ERROR_LOG_BUFFER;
values[0]  = (void*)error_log;
options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
values[1]  = (void*)BUFFER_SIZE;
options[2] = CU_JIT_BINARY_LOADER_THREAD_COUNT;
values[2]  = 0; // Use as many threads as CPUs on the machine
err = cuModuleLoadDataEx(&cuModule, cubinCode, 3, options, values);
if (err != CUDA_SUCCESS)
    printf("Link error:\n%s\n", error_log);
```

完整代码见 CUDA 示例 `ptxjit`。

## 3.3.3 Kernel 执行（Kernel Execution）

`cuLaunchKernel()` 使用给定的执行配置启动 Kernel。

参数可以通过两种方式传递：第一种方式是指针数组（`cuLaunchKernel()` 的倒数第二个参数），其中第 *n* 个指针对应第 *n* 个参数，并指向一段用于复制该参数的内存；第二种方式是使用额外选项（`cuLaunchKernel()` 的最后一个参数）。

当参数通过额外选项 `CU_LAUNCH_PARAM_BUFFER_POINTER` 传递时，应用程序需要传入一个单独缓冲区的指针，并确保缓冲区中的各参数按照设备代码中对应参数类型的对齐要求彼此正确偏移。

内置向量类型在设备代码中的对齐要求列于[表 43：设备代码中的向量类型对齐要求](../05-technical-appendices/cpp-language-extensions.html#vector-types-alignment-requirements-in-device-code)。对于其他基本类型，设备代码中的对齐要求与主机代码中的对齐要求一致，因此可以使用 `__alignof()` 获取。唯一的例外是：主机编译器可能会将 `double` 和 `long long`（以及 64 位系统上的 `long`）按照一个字对齐，而不是按照两个字对齐，例如使用 `gcc` 的编译选项 `-mno-align-double`；设备代码中这些类型始终按照两个字对齐。

`CUdeviceptr` 是整数，但它表示指针，因此其对齐要求是 `__alignof(void*)`。

下面的代码示例使用宏 `ALIGN_UP()` 调整每个参数的偏移量以满足其对齐要求，并使用宏 `ADD_TO_PARAM_BUFFER()` 将每个参数加入传递给 `CU_LAUNCH_PARAM_BUFFER_POINTER` 选项的参数缓冲区。

```cpp
#define ALIGN_UP(offset, alignment) \
      (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

char paramBuffer[1024];
size_t paramBufferSize = 0;

#define ADD_TO_PARAM_BUFFER(value, alignment)                   \
    do {                                                        \
        paramBufferSize = ALIGN_UP(paramBufferSize, alignment); \
        memcpy(paramBuffer + paramBufferSize,                   \
               &(value), sizeof(value));                        \
        paramBufferSize += sizeof(value);                       \
    } while (0)

int i;
ADD_TO_PARAM_BUFFER(i, __alignof(i));
float4 f4;
ADD_TO_PARAM_BUFFER(f4, 16); // float4's alignment is 16
char c;
ADD_TO_PARAM_BUFFER(c, __alignof(c));
float f;
ADD_TO_PARAM_BUFFER(f, __alignof(f));
CUdeviceptr devPtr;
ADD_TO_PARAM_BUFFER(devPtr, __alignof(devPtr));
float2 f2;
ADD_TO_PARAM_BUFFER(f2, 8); // float2's alignment is 8

void* extra[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, paramBuffer,
    CU_LAUNCH_PARAM_BUFFER_SIZE,    &paramBufferSize,
    CU_LAUNCH_PARAM_END
};
cuLaunchKernel(cuFunction,
               blockWidth, blockHeight, blockDepth,
               gridWidth, gridHeight, gridDepth,
               0, 0, 0, extra);
```

结构体的对齐要求等于其字段对齐要求中的最大值。因此，包含内置向量类型、`CUdeviceptr` 或未对齐的 `double` 和 `long long` 的结构体，其对齐要求可能在设备代码和主机代码之间不同。这类结构体的填充方式也可能不同。例如，下面的结构体在主机代码中完全没有填充，但在设备代码中，字段 `f` 后面会填充 12 个字节，因为字段 `f4` 的对齐要求是 16。

```cpp
typedef struct {
    float  f;
    float4 f4;
} myStruct;
```

## 3.3.4 Runtime API 与 Driver API 的互操作（Interoperability between Runtime and Driver APIs）

应用程序可以混合使用 Runtime API 代码和 Driver API 代码。

如果通过 Driver API 创建 context 并将其设为当前 context，那么后续的 Runtime API 调用会使用这个 context，而不是创建新的 context。

如果 Runtime 已初始化，可以使用 `cuCtxGetCurrent()` 获取初始化期间创建的 context。后续 Driver API 调用可以使用这个 context。

Runtime 隐式创建的 context 称为 primary context（见 [Runtime Initialization](../02-programming-gpus/intro-to-cuda-cpp.html#intro-cpp-runtime-initialization)）。可以通过 Driver API 的 [Primary Context Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html) 函数管理它。

设备内存可以使用任一 API 分配和释放。`CUdeviceptr` 可以转换为普通指针，反之亦然：

```cpp
CUdeviceptr devPtr;
float* d_data;

// Allocation using driver API
cuMemAlloc(&devPtr, size);
d_data = (float*)devPtr;

// Allocation using runtime API
cudaMalloc(&d_data, size);
devPtr = (CUdeviceptr)d_data;
```

这意味着，使用 Driver API 编写的应用程序可以调用使用 Runtime API 编写的库，例如 cuFFT、cuBLAS 等。

参考手册中设备管理和版本管理部分的所有函数都可以互换使用。
