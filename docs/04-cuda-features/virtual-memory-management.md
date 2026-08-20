---
title: 4.16 虚拟内存管理（Virtual Memory Management）
description: CUDA 虚拟内存管理 API、内存共享、映射、访问权限和高级配置。
---

# 4.16 虚拟内存管理（Virtual Memory Management）

在 CUDA 编程模型中，内存分配调用（例如 `cudaMalloc()`）会返回 GPU 内存中的地址。该地址可以与任意 CUDA API 一起使用，也可以在设备 kernel 中使用。开发者可以使用 `cudaEnablePeerAccess` 为这些内存分配启用对等设备访问，从而让不同设备上的 kernel 访问同一份数据。但是，所有过去和未来的用户分配也会映射到目标对等设备。这可能导致用户无意中承担将所有 `cudaMalloc` 分配映射到对等设备的运行时开销。在大多数情况下，应用只需与另一设备共享少量分配，通常没有必要将所有分配映射到所有设备。此外，将这种方法扩展到多节点设置本身也很困难。

CUDA 提供了*虚拟内存管理*（VMM）API，让开发者能够显式、低级地控制这一过程。

虚拟内存分配是由操作系统和内存管理单元（MMU）管理的复杂过程，分为两个关键阶段。首先，操作系统为程序保留一段连续的虚拟地址范围，但不分配任何物理内存。然后，当程序第一次尝试使用这段内存时，操作系统提交这些虚拟地址，并按需将物理存储分配给虚拟页。

CUDA 的 VMM API 将类似概念带到了 GPU 内存管理中：开发者可以显式保留一段虚拟地址范围，之后再将它映射到物理 GPU 内存。使用 VMM 时，应用可以明确选择哪些分配允许其他设备访问。

VMM API 使复杂应用能够更高效地管理多个 GPU（以及 CPU 核心）之间的内存。通过手动控制内存保留、映射和访问权限，VMM API 支持细粒度数据共享、零拷贝传输和自定义内存分配器等高级技术。CUDA VMM API 向用户提供了管理应用 GPU 内存所需的细粒度控制。

开发者可以从 VMM API 获得以下主要收益：

- 对虚拟内存和物理内存管理进行细粒度控制，允许将不连续的物理内存块分配并映射到连续的虚拟地址空间。这有助于减少 GPU 内存碎片并提高内存利用率，尤其适用于深度神经网络训练等大型工作负载。
- 通过将虚拟地址空间保留与物理内存分配分离，高效地分配和释放内存。开发者可以保留大的虚拟内存区域，并按需映射物理内存，无需代价高昂的内存复制或重新分配，从而改善动态数据结构和可变大小内存分配的性能。
- 可以动态扩大 GPU 内存分配，而无需复制并重新分配全部数据，类似 CPU 内存管理中的 `realloc` 或 `std::vector`。这支持更灵活、更高效的 GPU 内存使用模式。
- 通过提供低级 API，改善开发效率和应用性能，使开发者能够构建复杂的内存分配器和缓存管理系统，例如动态管理大语言模型中的键值缓存，从而提高吞吐量并降低延迟。
- CUDA VMM API 对分布式多 GPU 设置非常有价值，因为它可以在多个 GPU 之间高效共享和访问内存。通过将虚拟地址与物理内存解耦，该 API 允许开发者创建统一虚拟地址空间，并将数据动态映射到不同 GPU。这可以优化内存使用并降低数据传输开销。例如，NVIDIA 的 NCCL 和 NVSHMEM 等库都在积极使用 VMM。

总之，CUDA VMM API 为开发者提供了超越传统 malloc 类抽象的高级工具，用于进行精细、高效、灵活且可扩展的 GPU 内存管理；这对于高性能和大容量内存应用非常重要。

::: note

本节介绍的 API 集要求系统支持 UVA。请参阅 [虚拟内存管理 API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)。

:::

## 4.16.1 前置知识（Preliminaries）

### 4.16.1.1 定义（Definitions）

**Fabric 内存（Fabric Memory）：** Fabric 内存是指可以通过高速互连 fabric（例如 NVIDIA NVLink 和 NVSwitch）访问的内存。该 fabric 为多个 GPU 或节点提供内存一致性和高带宽通信层，使它们能够高效共享内存，就像内存连接在统一 fabric 上，而不是孤立地连接在各个设备上一样。

CUDA 12.4 及更高版本提供 VMM 分配句柄类型 `CU_MEM_HANDLE_TYPE_FABRIC`。在受支持的平台上，并且 NVIDIA IMEX 守护进程正在运行时，这种分配句柄类型不仅支持通过 MPI 等任意通信机制在节点内共享分配，也支持跨节点共享。这使多节点 NVLink 系统中的 GPU 可以映射同一 NVLink fabric 中其他 GPU 的内存，即使这些 GPU 位于不同节点上。

**内存句柄（Memory Handles）：** 在 VMM 中，句柄是不透明标识符，表示物理内存分配。这些句柄是底层 CUDA VMM API 管理内存的核心，提供了对可以映射到虚拟地址空间的物理内存对象的灵活控制。一个句柄唯一标识一个物理内存分配。句柄作为内存资源的抽象引用，不暴露直接指针；它们支持跨进程或设备导出、导入内存，从而实现内存共享和虚拟化。

**IMEX 通道（IMEX Channels）：** IMEX 是 *internode memory exchange*（节点间内存交换）的缩写，是 NVIDIA 用于跨不同节点进行 GPU-GPU 通信的解决方案的一部分。IMEX 通道是 GPU driver 的一项功能，为 IMEX 域中的多用户或多节点环境提供基于用户的内存隔离。IMEX 通道充当安全和隔离机制。

IMEX 通道与 fabric 句柄直接相关，在多节点 GPU 通信中必须启用。当一个 GPU 分配内存并希望让另一节点上的 GPU 访问这段内存时，它首先需要导出该内存的句柄。在导出过程中，IMEX 通道用于生成安全的 fabric 句柄；只有拥有正确通道访问权限的远程进程才能导入该句柄。

**单播内存访问（Unicast Memory Access）：** 在 VMM API 中，单播内存访问是指将物理内存受控、直接地映射到特定设备或进程的唯一虚拟地址范围。与向多个设备广播访问不同，单播内存访问意味着为某个 GPU 设备明确授予读写权限，使它能访问映射到物理内存分配的保留虚拟地址范围。

**多播内存访问（Multicast Memory Access）：** 在 VMM API 中，多播内存访问是指使用多播机制，将单个物理内存分配或区域同时映射到多个设备的虚拟地址空间。这允许数据以一对多的方式在多个 GPU 之间高效共享，减少重复数据传输并提高通信效率。NVIDIA CUDA VMM API 支持创建多播对象，将多个设备的物理内存分配绑定在一起。

### 4.16.1.2 查询支持情况（Query for Support）

应用在尝试使用功能之前应查询该功能是否受支持，因为可用性可能取决于 GPU 架构、driver 版本以及所使用的具体软件库。下面介绍如何以编程方式检查所需支持。

**VMM 支持：** 在尝试使用 VMM API 之前，应用必须确认计划使用的设备支持 CUDA 虚拟内存管理。下面的代码示例查询 VMM 支持情况：

```cpp
int deviceSupportsVmm;
CUresult result = cuDeviceGetAttribute(&deviceSupportsVmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device);
if (deviceSupportsVmm != 0) {
    // `device` 支持虚拟内存管理
}
```

**Fabric 内存支持：** 在尝试使用 fabric 内存之前，应用必须确认计划使用的设备支持 fabric 内存。下面的代码示例查询 fabric 内存支持情况：

```cpp
int deviceSupportsFabricMem;
CUresult result = cuDeviceGetAttribute(&deviceSupportsFabricMem, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device);
if (deviceSupportsFabricMem != 0) {
    // `device` 支持 Fabric 内存
}
```

除了使用 `CU_MEM_HANDLE_TYPE_FABRIC` 作为句柄类型，并且不需要使用操作系统原生机制在进程之间交换可共享句柄之外，使用 fabric 内存与使用其他分配句柄类型没有区别。

**IMEX 通道支持：** 在 IMEX 域内，IMEX 通道为多用户环境启用安全的内存共享。NVIDIA driver 通过创建字符设备 `nvidia-caps-imex-channels` 实现这一功能。要使用基于 fabric 句柄的共享，用户应验证以下两点：

- 首先，应用必须验证 `/proc/devices` 下存在该设备：

```text
# cat /proc/devices | grep nvidia
195 nvidia
195 nvidiactl
234 nvidia-caps-imex-channels
509 nvidia-nvswitch

nvidia-caps-imex-channels 设备应具有一个主设备号（例如 234）。
```

- 其次，两个 CUDA 进程（导出进程和导入进程）要共享内存，必须都能访问同一个 IMEX 通道文件。这些文件（例如 `/dev/nvidia-caps-imex-channels/channel0`）是表示单个 IMEX 通道的节点。系统管理员必须创建这些文件，例如使用 `mknod()` 命令。

```text
# mknod /dev/nvidia-caps-imex-channels/channelN c <major_number> 0

该命令使用从 /proc/devices 获取的主设备号创建 channelN。
```

::: note

默认情况下，如果指定了 `NVreg_CreateImexChannel0` 模块参数，driver 可以创建 channel0。

:::

**多播对象支持：** 在尝试使用多播对象之前，应用必须确认计划使用的设备支持多播对象。下面的代码示例查询多播对象支持情况：

```cpp
int deviceSupportsMultiCast;
CUresult result = cuDeviceGetAttribute(&deviceSupportsMultiCast, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device);
if (deviceSupportsMultiCast != 0) {
    // `device` 支持多播对象
}
```

## 4.16.2 API 概览（API Overview）

VMM API 为开发者提供了粒度很细的虚拟内存管理控制。VMM 是非常底层的 API，要求直接使用 [CUDA Driver API](../03-advanced-cuda/driver-api.html)。这个通用 API 既可用于单节点环境，也可用于多节点环境。

要有效使用 VMM，开发者必须掌握内存管理中的几个关键概念：

- 了解操作系统虚拟内存的基础知识，包括操作系统如何处理页和地址空间。
- 必须理解内存层次结构和硬件特性。
- 熟悉进程间通信（IPC）方法，例如 socket 或消息传递。
- 具备内存访问权限方面的基本安全知识。

![VMM 使用概览](/images/chapter-04/vmm-overview-diagram.png)

图 55：VMM 使用概览。该图概述了使用 VMM 所需的一系列步骤。流程从评估环境配置开始。根据评估结果，用户必须作出一个关键的初始决策：使用 fabric 内存句柄，还是使用操作系统特定句柄。根据句柄选择的不同，后续需要执行不同的一组步骤。但是，无论选择哪种句柄，最终的内存管理操作——具体包括映射、保留以及设置已分配内存的访问权限——都是相同的。

VMM API 工作流包含一系列内存管理步骤，重点是不同设备或进程之间的内存共享。首先，开发者必须在源设备上分配物理内存。为实现共享，VMM API 使用句柄向目标设备或进程传递必要信息。用户必须导出一个用于共享的句柄，可以是操作系统特定句柄，也可以是 fabric 特定句柄。操作系统特定句柄仅限于单节点内的进程间通信，而 fabric 特定句柄更灵活，可以用于单节点和多节点环境。请注意，使用 fabric 特定句柄需要启用 IMEX 通道。

导出句柄后，必须使用进程间通信协议将它共享给接收进程；具体采用哪种方法由开发者决定。接收进程随后使用 VMM API 导入句柄。句柄成功导出、共享并导入后，源进程和目标进程都必须保留虚拟地址空间，以便映射已分配的物理内存。最后一步是为每个设备设置内存访问权限，确保建立正确的权限。包含两种句柄方法的完整过程在随附图中进一步说明。

## 4.16.3 单播内存共享（Unicast Memory Sharing）

GPU 内存可以在一台具有多个 GPU 的机器上共享，也可以跨机器网络共享。过程包括以下步骤：

- **分配并导出：** 一个 GPU 上的 CUDA 程序分配内存，并获取该内存的可共享句柄。
- **共享并导入：** 通过 IPC、MPI、NCCL 等方式将句柄发送给节点上的其他程序。接收 GPU 上的 CUDA driver 导入句柄并创建所需的内存对象。
- **保留并映射：** driver 创建从程序虚拟地址（VA）到 GPU 物理地址（PA），再到其网络 Fabric 地址（FA）的映射。
- **访问权限：** 为分配设置访问权限。
- **释放内存：** 程序结束执行时释放所有分配。

![单播内存共享示例](/images/chapter-04/unicast-memory-sharing.png)

图 56：单播内存共享示例（Unicast Memory Sharing Example）

### 4.16.3.1 分配并导出（Allocate and Export）

**分配物理内存：** 使用虚拟内存管理 API 分配内存的第一步，是创建一个为分配提供后备存储的物理内存块。要分配物理内存，应用必须使用 `cuMemCreate` API。该函数创建的分配不包含任何设备或主机映射。函数参数 `CUmemGenericAllocationHandle` 描述要分配的内存属性，例如分配位置、是否要将分配共享给其他进程（或图形 API），以及要分配内存的物理属性。用户必须确保请求的分配大小按照适当粒度对齐。可以使用 `cuMemGetAllocationGranularity` 查询分配的粒度要求。

**Linux 操作系统特定句柄：**

```cpp
CUmemGenericAllocationHandle allocatePhysicalMemory(int device, size_t size) {
    CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleType = handleType;

    size_t granularity = 0;
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // 确保大小符合该分配的粒度要求
    size_t padded_size = ROUND_UP(size, granularity);

    // 分配物理内存
    CUmemGenericAllocationHandle allocHandle;
    cuMemCreate(&allocHandle, padded_size, &prop, 0);

    return allocHandle;
}
```

**Fabric 句柄：**

```cpp
CUmemGenericAllocationHandle allocatePhysicalMemory(int device, size_t size) {
    CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_FABRIC;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleType = handleType;

    size_t granularity = 0;
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // 确保大小符合该分配的粒度要求
    size_t padded_size = ROUND_UP(size, granularity);

    // 分配物理内存
    CUmemGenericAllocationHandle allocHandle;
    cuMemCreate(&allocHandle, padded_size, &prop, 0);

    return allocHandle;
}
```

::: note

`cuMemCreate` 分配的内存由它返回的 `CUmemGenericAllocationHandle` 引用。请注意，该引用不是指针，其内存当前还不可访问。

:::

::: note

可以使用 `cuMemGetAllocationPropertiesFromHandle` 查询分配句柄的属性。

:::

**导出内存句柄：** CUDA 虚拟内存管理 API 提供了一种使用句柄进行进程间通信的新机制，用于交换有关分配和物理地址空间的必要信息。可以导出操作系统特定 IPC 的句柄，也可以导出 fabric 特定 IPC 的句柄。操作系统特定 IPC 句柄只能用于单节点设置；fabric 特定句柄可用于单节点或多节点设置。

**Linux 操作系统特定句柄：**

```cpp
CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
CUmemGenericAllocationHandle handle = allocatePhysicalMemory(0, 1<<21);
int fd;
cuMemExportToShareableHandle(&fd, handle, handleType, 0);
```

**Fabric 句柄：**

```cpp
CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_FABRIC;
CUmemGenericAllocationHandle handle = allocatePhysicalMemory(0, 1<<21);
CUmemFabricHandle fh;
cuMemExportToShareableHandle(&fh, handle, handleType, 0);
```

::: note

操作系统特定句柄要求所有进程属于同一操作系统。

:::

::: note

Fabric 特定句柄要求系统管理员启用 IMEX 通道。

:::

可以使用 [memMapIpcDrv 示例](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features/memMapIPCDrv/) 了解如何将 IPC 用于 VMM 分配。

### 4.16.3.2 共享并导入（Share and Import）

**共享内存句柄：** 导出句柄后，必须使用进程间通信协议将其共享给接收进程。开发者可以自由选择任何共享句柄的方法；具体 IPC 方法取决于应用设计和环境。常见方法包括操作系统特定的进程间 socket 和分布式消息传递。使用操作系统特定 IPC 可以高性能地传输，但仅限于同一机器上的进程且不可移植。Fabric 特定 IPC 更简单、更可移植，但需要系统级支持。所选方法必须安全、可靠地将句柄数据传输到目标进程，使目标进程能够导入内存并建立有效映射。选择 IPC 方法的灵活性使 VMM API 能够集成到各种系统架构中，从单节点应用到分布式多节点设置都适用。下面的代码片段分别用 socket 编程和 MPI 展示共享及接收句柄。

**发送：操作系统特定 IPC（Linux）：**

```cpp
int ipcSendShareableHandle(int socket, int fd, pid_t process) {
    struct msghdr msg;
    struct iovec iov[1];

    union {
        struct cmsghdr cm;
        char* control;
    } control_un;

    size_t sizeof_control = CMSG_SPACE(sizeof(int)) * sizeof(char);
    control_un.control = (char*) malloc(sizeof_control);

    struct cmsghdr *cmptr;
    ssize_t readResult;
    struct sockaddr_un cliaddr;
    socklen_t len = sizeof(cliaddr);

    // 构造要发送此可共享句柄的客户端地址
    memset(&cliaddr, 0, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;
    char temp[20];
    sprintf(temp, "%s%u", "/tmp/", process);
    strcpy(cliaddr.sun_path, temp);
    len = sizeof(cliaddr);

    // 向客户端发送相应的可共享句柄
    int sendfd = fd;

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof_control;

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;

    memmove(CMSG_DATA(cmptr), &sendfd, sizeof(sendfd));

    msg.msg_name = (void *)&cliaddr;
    msg.msg_namelen = sizeof(struct sockaddr_un);

    iov[0].iov_base = (void *)"";
    iov[0].iov_len = 1;
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    ssize_t sendResult = sendmsg(socket, &msg, 0);
    if (sendResult <= 0) {
        perror("IPC failure: Sending data over socket failed");
        free(control_un.control);
        return -1;
    }

    free(control_un.control);
    return 0;
}
```

**发送：操作系统特定 IPC（Windows）：**

```cpp
int ipcSendShareableHandle(HANDLE *handle, HANDLE &shareableHandle, PROCESS_INFORMATION process) {
    HANDLE hProcess = OpenProcess(PROCESS_DUP_HANDLE, FALSE, process.dwProcessId);
    HANDLE hDup = INVALID_HANDLE_VALUE;
    DuplicateHandle(GetCurrentProcess(), shareableHandle, hProcess, &hDup, 0, FALSE, DUPLICATE_SAME_ACCESS);
    DWORD cbWritten;
    WriteFile(handle->hMailslot[i], &hDup, (DWORD)sizeof(hDup), &cbWritten, (LPOVERLAPPED)NULL);
    CloseHandle(hProcess);
    return 0;
}
```

**发送：Fabric IPC：**

```cpp
MPI_Send(&fh, sizeof(CUmemFabricHandle), MPI_BYTE, 1, 0, MPI_COMM_WORLD);
```

**接收：操作系统特定 IPC（Linux）：**

```cpp
int ipcRecvShareableHandle(int socket, int* fd) {
    struct msghdr msg = {0};
    struct iovec iov[1];
    struct cmsghdr cm;

    // 使用 union 确保控制数组满足对齐要求
    union {
        struct cmsghdr cm;
        // QNX 上不能使用此写法，因为 QNX 的 CMSG_SPACE 会调用 __cmsg_alignbytes，
        // 而 __cmsg_alignbytes 是运行时函数而非编译期宏
        // char control[CMSG_SPACE(sizeof(int))]
        char* control;
    } control_un;

    size_t sizeof_control = CMSG_SPACE(sizeof(int)) * sizeof(char);
    control_un.control = (char*) malloc(sizeof_control);
    struct cmsghdr *cmptr;
    ssize_t n;
    int receivedfd;
    char dummy_buffer[1];
    ssize_t sendResult;
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof_control;

    iov[0].iov_base = (void *)dummy_buffer;
    iov[0].iov_len = sizeof(dummy_buffer);

    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    if ((n = recvmsg(socket, &msg, 0)) <= 0) {
        perror("IPC failure: Receiving data over socket failed");
        free(control_un.control);
        return -1;
    }

    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) &&
        (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
        if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
        free(control_un.control);
        return -1;
        }

        memmove(&receivedfd, CMSG_DATA(cmptr), sizeof(receivedfd));
        *fd = receivedfd;
    } else {
        free(control_un.control);
        return -1;
    }

    free(control_un.control);
    return 0;
}
```

**接收：操作系统特定 IPC（Windows）：**

```cpp
int ipcRecvShareableHandle(HANDLE &handle, HANDLE *shareableHandle) {
    DWORD cbRead;
    ReadFile(handle, shareableHandle, (DWORD)sizeof(*shareableHandles), &cbRead, NULL);
    return 0;
}
```

**接收：Fabric IPC：**

```cpp
MPI_Recv(&fh, sizeof(CUmemFabricHandle), MPI_BYTE, 1, 0, MPI_COMM_WORLD);
```

**导入内存句柄：** 同样，用户可以导入操作系统特定 IPC 或 fabric 特定 IPC 的句柄。操作系统特定 IPC 句柄只能用于单节点；fabric 特定句柄可用于单节点或多节点。

**Linux 操作系统特定句柄：**

```cpp
CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
cuMemImportFromShareableHandle(handle, (void*) &fd, handleType);
```

**Fabric 句柄：**

```cpp
CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_FABRIC;
cuMemImportFromShareableHandle(handle, (void*) &fh, handleType);
```

### 4.16.3.3 保留并映射（Reserve and Map）

**保留虚拟地址范围：**

由于 VMM 将地址与内存区分开，应用必须划出一段地址范围，用于容纳 `cuMemCreate` 创建的内存分配。保留的地址范围至少要与用户计划放置其中的所有物理内存分配大小之和相同。

应用可以向 `cuMemAddressReserve` 传递适当参数来保留虚拟地址范围。获得的地址范围不关联任何设备或主机物理内存。保留的虚拟地址范围可以映射到系统中任意设备所属的内存块，从而为应用提供由不同设备内存提供后备并映射的连续 VA 范围。应用应使用 `cuMemAddressFree` 将虚拟地址范围归还给 CUDA。调用 `cuMemAddressFree` 前，用户必须确保整个 VA 范围都已取消映射。这些函数在概念上类似于 Linux 上的 `mmap` 和 `munmap`，或 Windows 上的 `VirtualAlloc` 和 `VirtualFree`。下面的代码片段展示该函数的用法：

```cpp
CUdeviceptr ptr;
// `ptr` 保存所保留虚拟地址范围的起始地址。
CUresult result = cuMemAddressReserve(&ptr, size, 0, 0, 0); // alignment = 0 表示默认对齐
```

**映射内存：** 前两节分别分配的物理内存和划出的虚拟地址空间体现了 VMM API 引入的内存与地址分离。要使已分配内存可用，用户必须将内存映射到地址空间。必须使用 `cuMemMap` 将从 `cuMemAddressReserve` 获得的地址范围与从 `cuMemCreate` 或 `cuMemImportFromShareableHandle` 获得的物理分配关联起来。

只要划出的地址空间足够，用户就可以将多个设备的分配关联到连续的虚拟地址范围。要解除物理分配与地址范围的关联，用户必须使用 `cuMemUnmap` 取消该映射地址。用户可以任意多次将内存映射到同一地址范围并取消映射，但必须确保不会尝试在已经映射的 VA 范围保留区上创建映射。下面的代码片段展示该函数的用法：

```cpp
CUdeviceptr ptr;
// `ptr`：之前由 cuMemAddressReserve 保留的地址范围中的地址。
// `allocHandle`：之前调用 cuMemCreate 获得的 CUmemGenericAllocationHandle。
CUresult result = cuMemMap(ptr, size, 0, allocHandle, 0);
```

### 4.16.3.4 访问权限（Access Rights）

CUDA 的虚拟内存管理 API 允许应用使用访问控制机制显式保护 VA 范围。使用 `cuMemMap` 将分配映射到地址范围的某个区域，并不会使该地址可访问；如果 CUDA kernel 访问它，程序会崩溃。用户必须在源设备和访问设备上使用 `cuMemSetAccess` 明确选择访问控制。该函数允许或限制特定设备访问已映射的地址范围。下面的代码片段展示该函数的用法：

```cpp
void setAccessOnDevice(int device, CUdeviceptr ptr, size_t size) {
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // 使地址可访问
    cuMemSetAccess(ptr, size, &accessDesc, 1);
}
```

VMM 提供的访问控制机制允许用户明确指定要与系统中其他对等设备共享哪些分配。如前所述，`cudaEnablePeerAccess` 会强制将使用 `cudaMalloc` 创建的所有过去和未来分配映射到目标对等设备。这在许多情况下很方便，因为用户不必跟踪系统中每个分配到每个设备的映射状态，但[这种方法会带来性能影响](https://devblogs.nvidia.com/introducing-low-level-gpu-virtual-memory-management/)。VMM 在分配粒度上提供访问控制，因此可以以最小开销执行对等映射。

[vectorAddMMAP 示例](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/vectorAddMMAP)可作为使用虚拟内存管理 API 的参考。

### 4.16.3.5 释放内存（Releasing the Memory）

要释放已分配的内存和地址空间，源进程和目标进程都应按以下顺序使用 `cuMemUnmap`、`cuMemRelease` 和 `cuMemAddressFree`。`cuMemUnmap` 将之前映射的内存区域从地址范围中取消映射，有效地将物理内存从保留的虚拟地址空间分离。接着，`cuMemRelease` 释放之前创建的物理内存，将其归还给系统。最后，`cuMemAddressFree` 释放之前保留的虚拟地址范围，使其可以供将来使用。这个特定顺序确保物理内存和虚拟地址空间都得到干净、完整的释放。

```cpp
cuMemUnmap(ptr, size);
cuMemRelease(handle);
cuMemAddressFree(ptr, size);
```

::: note

在操作系统特定句柄场景中，必须使用 `fclose` 关闭导出的句柄。Fabric 句柄场景不适用这一步。

:::

## 4.16.4 多播内存共享（Multicast Memory Sharing）

[多播对象管理 API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MULTICAST.html#group__CUDA__MULTICAST) 提供了创建多播对象的方法；结合上文介绍的[虚拟内存管理 API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)，应用可以在受支持的、通过 NVSwitch 连接的 NVLink GPU 上利用 NVLink SHARP。NVLink SHARP 允许 CUDA 应用利用 fabric 内计算，加速 NVSwitch 连接 GPU 之间的广播和归约等操作。为实现这一点，多个 NVLink 连接 GPU 组成一个多播团队，团队中的每个 GPU 都用物理内存为多播对象提供后备。因此，一个包含 N 个 GPU 的多播团队拥有多播对象的 N 个物理副本，每个副本位于一个参与 GPU 的本地内存中。使用多播对象映射的 [multimem PTX 指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem)会作用于多播对象的所有副本。

要使用多播对象，应用需要：

- 查询多播支持。
- 使用 `cuMulticastCreate` 创建多播句柄。
- 将多播句柄共享给控制应参与多播团队的 GPU 的所有进程。这一步使用上文介绍的 `cuMemExportToShareableHandle` 完成。
- 使用 `cuMulticastAddDevice` 添加所有应参与多播团队的 GPU。
- 对于每个参与 GPU，将使用 `cuMemCreate` 分配的物理内存绑定到多播句柄。必须先将所有设备添加到多播团队，然后才能在任意设备上绑定内存。
- 像普通单播映射那样保留地址范围、映射多播句柄并设置访问权限。物理内存可以同时存在单播和多播映射。有关如何保证同一物理内存的多个映射之间保持一致，请参阅[虚拟别名支持](#41653-虚拟别名支持)一节。
- 使用带多播映射的 [multimem PTX 指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem)。

[Multi GPU Programming Models](https://github.com/NVIDIA/multi-gpu-programming-models/) GitHub 仓库中的 `multi_node_p2p` 示例包含一个使用 fabric 内存（包括多播对象）利用 NVLink SHARP 的完整示例。请注意，该示例面向 NCCL 或 NVSHMEM 等库的开发者，展示了 NVSHMEM 等高级编程模型在（多节点）NVLink 域内部的工作方式。应用开发者通常应使用更高级的 MPI、NCCL 或 NVSHMEM 接口，而不是直接使用本 API。

### 4.16.4.1 分配多播对象（Allocating Multicast Objects）

可以使用 `cuMulticastCreate` 创建多播对象：

```cpp
CUmemGenericAllocationHandle createMCHandle(int numDevices, size_t size) {
    CUmemAllocationProp mcProp = {};
    mcProp.numDevices = numDevices;
    mcProp.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC; // 单节点也可以使用 CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

    size_t granularity = 0;
    cuMulticastGetGranularity(&granularity, &mcProp, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // 确保大小符合该分配的粒度要求
    size_t padded_size = ROUND_UP(size, granularity);

    mcProp.size = padded_size;

    // 创建多播对象；此时尚未关联设备和物理内存
    CUmemGenericAllocationHandle mcHandle;
    cuMulticastCreate(&mcHandle, &mcProp);

    return mcHandle;
}
```

### 4.16.4.2 向多播对象添加设备（Add Devices to Multicast Objects）

可以使用 `cuMulticastAddDevice` 将设备添加到多播团队：

```cpp
cuMulticastAddDevice(&mcHandle, device);
```

在任意设备的内存绑定到多播对象之前，必须在控制参与多播团队设备的所有进程中完成这一步。

### 4.16.4.3 将内存绑定到多播对象（Bind Memory to Multicast Objects）

创建多播对象并将所有参与设备添加到多播对象后，必须为每个设备使用 `cuMemCreate` 分配物理内存，为多播对象提供后备：

```cpp
cuMulticastBindMem(mcHandle, mcOffset, memHandle, memOffset, size, 0 /*flags*/);
```

### 4.16.4.4 使用多播映射（Use Multicast Mappings）

要在 CUDA C++ 中使用多播映射，必须通过内联 PTX 使用 [multimem PTX 指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem)：

```cpp
__global__ void all_reduce_norm_barrier_kernel(float* l2_norm,
                                               float* partial_l2_norm_mc,
                                               unsigned int* arrival_counter_uc, unsigned int* arrival_counter_mc,
                                               const unsigned int expected_count) {
    assert( 1 == blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z );
    float l2_norm_sum = 0.0;
#if __CUDA_ARCH__ >= 900

    // 对所有副本执行原子归约
    // 概念上可以看作 __threadfence_system(); atomicAdd_system(arrival_counter_mc, 1);
    cuda::ptx::multimem_red(cuda::ptx::release_t, cuda::ptx::scope_sys_t, cuda::ptx::op_add_t, arrival_counter_mc, n);

    // 对同一内存 arrival_counter_uc 和 arrival_counter_mc 的 Multicast（mc）
    // 与 Unicast（uc）访问之间需要 fence：
    // - fence.proxy 指令在可能通过不同 proxy 发生的内存访问之间建立顺序；
    // - .proxykind 限定符的 .alias 值表示通过虚拟别名地址对同一内存位置执行的访问。
    // 参见 https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar
    cuda::ptx::fence_proxy_alias();

    // 使用 acquire 顺序在 UC 映射上自旋等待，直到所有对等端到达本次迭代
    // 注意：所有 rank 必须在此 kernel 后到达另一个屏障，以免某个 rank 较慢时，
    // 下一次迭代中某个 rank 的到达解除本次屏障。
    cuda::atomic_ref<unsigned int,cuda::thread_scope_system> ac(arrival_counter_uc);
    while (expected_count > ac.load(cuda::memory_order_acquire));

    // 从所有副本执行原子加载归约。它不提供排序，因此可以使用 relaxed。
    asm volatile ("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];" : "=f"(l2_norm_sum) : "l"(partial_l2_norm_mc) : "memory");

#else
    #error "ERROR: multimem instructions require compute capability 9.0 or larger."
#endif

    *l2_norm = std::sqrt(l2_norm_sum);
}
```

## 4.16.5 高级配置（Advanced Configuration）

### 4.16.5.1 内存类型（Memory Type）

VMM 还提供一种机制，使应用可以分配某些设备支持的特殊类型内存。通过 `cuMemCreate`，应用可以使用 `CUmemAllocationProp::allocFlags` 指定内存类型要求，从而选择启用特定内存功能。应用必须确保请求的内存类型受设备支持。

### 4.16.5.2 可压缩内存（Compressible Memory）

可压缩内存可用于加速访问具有非结构化稀疏性以及其他可压缩数据模式的数据。根据数据情况，压缩可以节省 DRAM 带宽、L2 读取带宽和 L2 容量。希望在支持计算数据压缩的设备上分配可压缩内存的应用，可以将 `CUmemAllocationProp::allocFlags::compressionType` 设置为 `CU_MEM_ALLOCATION_COMP_GENERIC`。用户必须使用 `CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED` 查询设备是否支持计算数据压缩。下面的代码片段展示如何使用 `cuDeviceGetAttribute` 查询可压缩内存支持：

```cpp
int compressionSupported = 0;
cuDeviceGetAttribute(&compressionSupported, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, device);
```

在支持计算数据压缩的设备上，用户必须在分配时选择启用压缩，如下所示：

```cpp
prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
```

由于硬件资源有限等多种原因，分配可能不具备压缩属性。要验证标志是否生效，用户应使用 `cuMemGetAllocationPropertiesFromHandle` 查询已分配内存的属性。

```cpp
CUmemAllocationProp allocationProp = {};
cuMemGetAllocationPropertiesFromHandle(&allocationProp, allocationHandle);

if (allocationProp.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC)
{
    // 获得了可压缩内存分配
}
```

### 4.16.5.3 虚拟别名支持（Virtual Aliasing Support）

虚拟内存管理 API 允许通过多次调用 `cuMemMap` 并使用不同虚拟地址，为同一个分配创建多个虚拟内存映射或“代理（proxy）”。这称为虚拟别名。除非 PTX ISA 另有说明，否则在执行写入的设备操作（grid 启动、memcpy、memset 等）完成之前，对分配的一个代理执行的写入，都被认为与同一内存的其他代理不一致且不具备一致性。在写入设备操作之前已存在于 GPU 上、但在写入设备操作完成后才读取的 grid，其代理同样被认为不一致且不具备一致性。

例如，假设设备指针 A 和 B 是同一内存分配的虚拟别名，下面的代码片段行为未定义：

```cpp
__global__ void foo(char *A, char *B) {
  *A = 0x1;
  printf("%d\n", *B);    // 行为未定义！*B 可能取前一个值，
// 或者取介于两者之间的某个值。
}
```

如果这两个 kernel 按单调顺序排列（通过流或事件排序），下面的行为是有定义的：

```cpp
__global__ void foo1(char *A) {
  *A = 0x1;
}

__global__ void foo2(char *B) {
  printf("%d\n", *B);    // *B == *A == 0x1，假设 foo2 等待 foo1
// 完成后才启动。
}

cudaMemcpyAsync(B, input, size, stream1);    // 允许在操作边界处使用别名
// 让 foo1 可以访问 A。
foo1<<<1,1,0,stream1>>>(A);
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event);
foo2<<<1,1,0,stream2>>>(B);
cudaStreamWaitEvent(stream3, event);
cudaMemcpyAsync(output, B, size, stream3);  // foo2 和 cudaMemcpy 都读取，
                                            // 二者都等待执行写入的 foo1
                                            // 完成后再继续。
```

如果同一 kernel 必须通过不同代理访问同一分配，可以在两次访问之间使用 `fence.proxy.alias`。因此，上面的示例可以通过内联 PTX 汇编变为合法代码：

```cpp
__global__ void foo(char *A, char *B) {
  *A = 0x1;
  cuda::ptx::fence_proxy_alias();
  printf("%d\n", *B);    // *B == *A == 0x1
}
```

### 4.16.5.4 用于 IPC 的操作系统特定句柄细节（OS-Specific Handle Details for IPC）

使用 `cuMemCreate` 时，用户可以在分配时表明某个分配将用于进程间通信或图形互操作。应用可以将 `CUmemAllocationProp::requestedHandleTypes` 设置为平台特定字段来实现这一点。在 Windows 上，当 `CUmemAllocationProp::requestedHandleTypes` 设置为 `CU_MEM_HANDLE_TYPE_WIN32` 时，应用还必须在 `CUmemAllocationProp::win32HandleMetaData` 中指定一个 LPSECURITYATTRIBUTES 属性。该安全属性定义了导出分配可以传输给哪些其他进程。

用户必须在尝试导出使用 `cuMemCreate` 分配的内存之前，查询所请求句柄类型是否受支持。下面的代码片段展示了如何以平台特定方式查询句柄类型支持：

```cpp
int deviceSupportsIpcHandle;
#if defined(__linux__)
    cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
#else
    cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, device));
#endif
```

用户应按如下方式适当设置 `CUmemAllocationProp::requestedHandleTypes`：

```cpp
#if defined(__linux__)
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_WIN32;
    prop.win32HandleMetaData = // Windows 专用 LPSECURITYATTRIBUTES 属性。
#endif
```
