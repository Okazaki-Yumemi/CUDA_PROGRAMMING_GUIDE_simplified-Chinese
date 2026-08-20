---
title: 4.20 驱动入口点访问（Driver Entry Point Access）
description: 获取 CUDA Driver API 入口点、版本化函数指针以及运行时功能。
---

# 4.20 驱动入口点访问（Driver Entry Point Access）

## 4.20.1 简介（Introduction）

`Driver Entry Point Access` API 提供了获取 CUDA 驱动函数地址的方法。从 CUDA 11.3 开始，用户可以通过这些 API 获得函数指针，并调用可用的 CUDA Driver API。

这些 API 提供的功能类似于 POSIX 平台上的 `dlsym` 和 Windows 上的 `GetProcAddress`。通过这些 API，用户可以：

- 使用 CUDA Driver API 获取驱动函数的地址。
- 使用 CUDA Runtime API 获取驱动函数的地址。
- 请求 CUDA 驱动函数的*每线程默认流（per-thread default stream）*版本。详见[获取每线程默认流版本](#42033-获取每线程默认流版本retrieve-per-thread-default-stream-versions)。
- 在使用较旧工具包、但驱动程序较新的情况下访问新的 CUDA 功能。

## 4.20.2 驱动函数 typedef（Driver Function Typedefs）

为帮助获取 CUDA Driver API 入口点，CUDA Toolkit 提供了包含所有 CUDA 驱动 API 函数指针定义的头文件。这些头文件随 CUDA Toolkit 安装，并位于工具包的 `include/` 目录中。下表总结了各 CUDA API 头文件对应的、包含 `typedef` 的头文件。

| API 头文件 | API typedef 头文件 |
| --- | --- |
| `cuda.h` | `cudaTypedefs.h` |
| `cudaGL.h` | `cudaGLTypedefs.h` |
| `cudaProfiler.h` | `cudaProfilerTypedefs.h` |
| `cudaVDPAU.h` | `cudaVDPAUTypedefs.h` |
| `cudaEGL.h` | `cudaEGLTypedefs.h` |
| `cudaD3D9.h` | `cudaD3D9Typedefs.h` |
| `cudaD3D10.h` | `cudaD3D10Typedefs.h` |
| `cudaD3D11.h` | `cudaD3D11Typedefs.h` |

表 27：CUDA Driver API 的 typedef 头文件

上述头文件本身不定义实际的函数指针，而是定义函数指针的 typedef。例如，`cudaTypedefs.h` 中包含 Driver API `cuMemAlloc` 的以下 typedef：

```cpp
typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v3020)(CUdeviceptr_v2 *dptr, size_t bytesize);
typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v2000)(CUdeviceptr_v1 *dptr, unsigned int bytesize);
```

CUDA 驱动符号采用基于版本的命名方案：名称中通常带有 `_v*` 后缀，第一版例外。当某个 CUDA Driver API 的签名或语义发生变化时，会递增对应驱动符号的版本号。以 `cuMemAlloc` 为例，第一个驱动符号名是 `cuMemAlloc`，下一个符号名是 `cuMemAlloc_v2`。第一版 typedef 在 CUDA 2.0（2000）中引入，名称为 `PFN_cuMemAlloc_v2000`；下一版 typedef 在 CUDA 3.2（3020）中引入，名称为 `PFN_cuMemAlloc_v3020`。

使用这些 `typedef`，可以更容易地在代码中定义正确类型的函数指针：

```cpp
PFN_cuMemAlloc_v3020 pfn_cuMemAlloc_v2;
PFN_cuMemAlloc_v2000 pfn_cuMemAlloc_v1;
```

## 4.20.3 驱动函数获取（Driver Function Retrieval）

使用 Driver Entry Point Access API 和适当的 typedef，可以获取任意 CUDA Driver API 的函数指针。

### 4.20.3.1 使用 Driver API（Using the Driver API）

Driver API 要求传入 CUDA 版本，以便为请求的驱动符号获取 ABI 兼容版本。CUDA Driver API 为每个函数提供 ABI 版本，函数名通过 `_v*` 后缀表示版本。以 `cudaTypedefs.h` 中 `cuStreamBeginCapture` 的版本及其对应 typedef 为例：

```cpp
// cuda.h
CUresult CUDAAPI cuStreamBeginCapture(CUstream hStream);
CUresult CUDAAPI cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode);

// cudaTypedefs.h
typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10000)(CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10010)(CUstream hStream, CUstreamCaptureMode mode);
```

从上面的 typedef 可以看出，后缀 `_v10000` 和 `_v10010` 表示这些 API 分别在 CUDA 10.0 和 CUDA 10.1 中引入。

```cpp
#include <cudaTypedefs.h>

// 声明 cuStreamBeginCapture 的入口点
PFN_cuStreamBeginCapture_v10000 pfn_cuStreamBeginCapture_v1;
PFN_cuStreamBeginCapture_v10010 pfn_cuStreamBeginCapture_v2;

// 获取 cuStreamBeginCapture 驱动符号的函数指针
cuGetProcAddress("cuStreamBeginCapture", &pfn_cuStreamBeginCapture_v1, 10000, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
// 获取 cuStreamBeginCapture_v2 驱动符号的函数指针
cuGetProcAddress("cuStreamBeginCapture", &pfn_cuStreamBeginCapture_v2, 10010, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
```

如上面的代码所示，要获取驱动 API `cuStreamBeginCapture` 的 `_v1` 版本地址，CUDA 版本参数必须精确为 10.0（10000）。同理，要获取该 API 的 `_v2` 版本地址，CUDA 版本参数应为 10.1（10010）。使用更高的 CUDA 版本来获取某个特定版本的驱动 API，可能并不具备可移植性。例如，此处使用 11030 仍会返回 `_v2` 符号；但如果假设 CUDA 11.3 发布了 `_v3` 版本，那么在 CUDA 11.3 驱动下传入 CUDA 11.3 时，`cuGetProcAddress` 会开始返回更新的 `_v3` 符号。由于 `_v2` 与 `_v3` 符号的 ABI 和函数签名可能不同，使用为 `_v2` 符号准备的 `_v10010` typedef 调用 `_v3` 函数会产生未定义行为。

注意，使用无效的 CUDA 版本请求驱动 API 时，会返回 `CUDA_ERROR_NOT_FOUND` 错误。在上述代码示例中，传入小于 10000（CUDA 10.0）的版本都是无效的。

### 4.20.3.2 使用 Runtime API（Using the Runtime API）

Runtime API `cudaGetDriverEntryPointByVersion` 使用提供的 CUDA 版本，以与 `cuGetProcAddress` 相同的方式获取请求驱动符号的 ABI 兼容版本。在下面的代码示例中，所需的最低 CUDA 版本是 CUDA 11.2，因为 `cuMemAllocAsync` 正是在该版本中引入的。

```cpp
#include <cudaTypedefs.h>

int cudaVersion;
// 确保已安装 CUDA >= 11.2 的驱动，否则 cuGetProcAddress 会返回错误
status = cuDriverGetVersion(&cudaVersion);
if (cudaVersion >= 11020) {

   // 声明入口点
   PFN_cuMemAllocAsync_v11020 pfn_cuMemAllocAsync;

   // 初始化入口点
   cudaGetDriverEntryPointByVersion("cuMemAllocAsync", &pfn_cuMemAllocAsync, 11020, cudaEnableDefault, &driverStatus);

   // 调用入口点
   if(driverStatus == cudaDriverEntryPointSuccess && pfn_cuMemAllocAsync) {
       pfn_cuMemAllocAsync(...);
   }
}
```

### 4.20.3.3 获取每线程默认流版本（Retrieve Per-thread Default Stream Versions）

某些 CUDA Driver API 可以配置为使用*默认流（default stream）*或*每线程默认流（per-thread default stream）*语义。使用每线程默认流语义的 Driver API，其名称带有 `_ptsz` 或 `_ptds` 后缀。例如，`cuLaunchKernel` 有一个名为 `cuLaunchKernel_ptsz` 的每线程默认流变体。使用 Driver Entry Point Access API 时，用户可以请求 `cuLaunchKernel` 的每线程默认流版本，而不是默认流版本。将 CUDA Driver API 配置为默认流或每线程默认流语义会影响同步行为。更多细节请参阅 [CUDA Driver API 流同步行为](https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream)。

可以通过以下方式之一获取 Driver API 的默认流或每线程默认流版本：

- 使用编译选项 `--default-stream per-thread`，或定义宏 `CUDA_API_PER_THREAD_DEFAULT_STREAM`，以获得每线程默认流行为。
- 分别使用 `CU_GET_PROC_ADDRESS_LEGACY_STREAM/cudaEnableLegacyStream` 或 `CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM/cudaEnablePerThreadDefaultStream` 标志，强制使用默认流或每线程默认流行为。

### 4.20.3.4 访问新的 CUDA 功能（Access New CUDA features）

要访问新的 CUDA 驱动功能，始终建议安装最新的 CUDA Toolkit；但如果用户出于某种原因不希望更新，或者无法使用最新工具包，那么只更新 CUDA 驱动也可以通过这些 API 访问新的 CUDA 功能。下面假设用户使用 CUDA 12.3，并希望使用 CUDA 12.5 驱动中提供的新驱动 API `cuFoo`。以下代码示例演示了这种用法：

```cpp
int main()
{
    // 手动定义原型，因为 CUDA 12.3 的 cudaTypedefs.h 中没有 cuFoo typedef
    typedef CUresult (CUDAAPI *PFN_cuFoo_v12050)(...);
    PFN_cuFoo_v12050 pfn_cuFoo = NULL;
    CUdriverProcAddressQueryResult driverStatus;
    int cudaVersion;

    // 确保已安装 CUDA >= 12.5 的驱动，否则 cuGetProcAddress 会返回错误
    CUresult status = cuDriverGetVersion(&cudaVersion);
    if (cudaVersion >= 12050) {
        // 使用 cuGetProcAddress 获取 cuFoo API 的地址。
        // 由于 cuFoo 在 CUDA 12.5 中引入，指定 CUDA 版本为 12050
        CUresult status = cuGetProcAddress("cuFoo", &pfn_cuFoo, 12050, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);

        if (status == CUDA_SUCCESS && pfn_cuFoo) {
            pfn_cuFoo(...);
        }
        else {
            printf("Cannot retrieve the address to cuFoo - driverStatus = %d\n", driverStatus);
            assert(0);
        }
    }

    // 此处为其余代码
}
```

下一个示例讨论如何获取 CUDA Toolkit 某个次要版本中发布的新 API 版本。注意，在 `cuda.h` 头文件中，只有到达主要版本边界时，才会通过版本宏将 `cuDeviceGetUuid` 提升为 `_v2`。因此，在 11.4 及之后的版本中，可以使用下面的示例获取 `_v2` 版本。

注意，在这种情况下，原始版本（而不是 `_v2` 版本）的 typedef 如下：

```cpp
typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid_v9020)(CUuuid *uuid, CUdevice_v1 dev);
```

而 `_v2` 版本的 typedef 如下：

```cpp
typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid_v11040)(CUuuid *uuid, CUdevice_v1 dev);
```

```cpp
#include <cudaTypedefs.h>

CUuuid uuid;
CUdevice dev;
CUresult status;
int cudaVersion;
CUdriverProcAddressQueryResult driverStatus;

status = cuDeviceGet(&dev, 0); // 获取设备 0
// 处理 status

// 确保已安装 CUDA >= 11.4 的驱动，否则 cuGetProcAddress 会返回错误
status = cuDriverGetVersion(&cudaVersion);
if (cudaVersion >= 11040) {
   PFN_cuDeviceGetUuid_v11040 pfn_cuDeviceGetUuid;
   status = cuGetProcAddress("cuDeviceGetUuid", &pfn_cuDeviceGetUuid, 11040, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
   if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuid) {
      pfn_cuDeviceGetUuid(&uuid, dev);
   }
}
```

## 4.20.4 `cuGetProcAddress` 使用指南（Guidelines for cuGetProcAddress）

使用 `cuGetProcAddress` 时，请注意以下指南：

- 将传给 `cuGetProcAddress` 的 CUDA 版本编码为与 typedef 版本匹配的值（不要使用编译期常量，例如 `CUDA_VERSION`；也不要使用动态版本，例如 `cuDriverGetVersion` 返回的版本）。
- 在调用 `cuGetProcAddress` 前，检查当前驱动版本（例如通过 `cuDriverGetVersion` 获取）是否足够；否则预期会发生错误，或者可能返回意外的符号。

### 4.20.4.1 Runtime API 使用指南（Guidelines for Runtime API Usage）

除非另有说明，CUDA Runtime API `cudaGetDriverEntryPointByVersion` 的使用指南与驱动入口点 `cuGetProcAddress` 类似，因为它允许用户请求特定的 CUDA 驱动版本。

## 4.20.5 确定 `cuGetProcAddress` 失败原因（Determining cuGetProcAddress Failure Reasons）

`cuGetProcAddress` 有两类错误：（1）API/用法错误；（2）无法找到所请求的 Driver API。第一类错误会通过 `CUresult` 返回值传递 API 的错误码，例如将 `NULL` 作为 `pfn` 变量传入，或传入无效的 `flags`。

第二类错误会编码在 `CUdriverProcAddressQueryResult` 类型的 `*symbolStatus` 中，可用于区分驱动无法找到所请求符号时的潜在原因。下面是一个示例：

```cpp
// cuDeviceGetExecAffinitySupport 在 CUDA 11.4 中引入
#include <cuda.h>
CUdriverProcAddressQueryResult driverStatus;
cudaVersion = ...;
status = cuGetProcAddress("cuDeviceGetExecAffinitySupport", &pfn, cudaVersion, 0, &driverStatus);
if (CUDA_SUCCESS == status) {
    if (CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT == driverStatus) {
        printf("We can use the new feature when you upgrade cudaVersion to 11.4, but CUDA driver is good to go!\n");
        // 表示 cudaVersion < 11.4，但运行时使用的是 CUDA >= 11.4 的驱动
    }
    else if (CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND == driverStatus) {
        printf("Please update both CUDA driver and cudaVersion to at least 11.4 to use the new feature!\n");
        // 表示驱动 < 11.4；由于没有找到字符串，cudaVersion 的值无关紧要
    }
    else if (CU_GET_PROC_ADDRESS_SUCCESS == driverStatus && pfn) {
        printf("You're using cudaVersion and CUDA driver >= 11.4, using new feature!\n");
        pfn();
    }
}
```

第一种情况的返回码为 `CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT`，表示在 CUDA 驱动中搜索时找到了 `symbol`，但它是在所提供的 `cudaVersion` 之后才添加的。在上面的示例中，当运行在 CUDA 11.4 或更高版本的驱动上时，如果指定的 `cudaVersion` 为 11030 或更低，就会得到 `CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT`。这是因为 `cuDeviceGetExecAffinitySupport` 在 CUDA 11.4（11040）中才加入。

第二种情况的返回码为 `CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND`，表示在 CUDA 驱动中搜索时没有找到 `symbol`。这可能有多种原因，例如旧驱动不支持该 CUDA 函数，或者函数名存在拼写错误。后者与上一个示例类似：如果用户将 `symbol` 写成 `CUDeviceGetExecAffinitySupport`（注意字符串开头是大写 `CU`），`cuGetProcAddress` 就无法找到该 API，因为字符串不匹配。前一种情况的例子是：用户针对支持新 API 的 CUDA 驱动开发应用，却将应用部署到较旧的 CUDA 驱动上。仍以最后一个示例为例，如果开发者使用 CUDA 11.4 或更高版本开发，但部署环境使用 CUDA 11.3 驱动，那么开发期间 `cuGetProcAddress` 可能成功；而部署到 CUDA 11.3 驱动后，该调用将不再成功，并会在 `driverStatus` 中返回 `CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND`。
