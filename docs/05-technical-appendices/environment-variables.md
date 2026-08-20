---
title: 5.2 CUDA 环境变量（CUDA Environment Variables）
description: CUDA 环境变量参考
---

<a id="cuda-environment-variables"></a>

# 5.2 CUDA 环境变量（CUDA Environment Variables）

以下各节列出了 CUDA 环境变量。与多进程服务（Multi-Process Service，MPS）相关的变量记录在[GPU Deployment and Management Guide](https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html#environment-variables)中。

<a id="device-enumeration-and-properties"></a>

## 5.2.1 设备枚举与属性（Device Enumeration and Properties）

<a id="cuda-visible-devices"></a>

### 5.2.1.1 `CUDA_VISIBLE_DEVICES`

该环境变量控制 CUDA 应用可见的 GPU 设备及其枚举顺序。

- 如果未设置该变量，则所有 GPU 设备均可见。

- 如果将该变量设置为空字符串，则没有 GPU 设备可见。

**可能的取值**：以逗号分隔的 GPU 标识符序列。

GPU 标识符可以采用以下形式：

- **整数索引**：对应于系统中 GPU 的序号，该序号由从 0 开始的[nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html)确定。例如，设置 `CUDA_VISIBLE_DEVICES=2,1` 会使设备 0 不可见，并先枚举设备 2、再枚举设备 1。

  - 如果遇到无效索引，则列表中出现在该无效索引之前的设备仍然可见。例如，设置 `CUDA_VISIBLE_DEVICES=0,2,-1,1` 会使设备 0 和 2 可见；设备 1 位于无效索引 `-1` 之后，因此不可见。

- **GPU UUID 字符串**：其格式应与 `nvidia-smi -L` 输出的格式相同，例如 `GPU-8932f937-d72c-4106-c12f-20bd9faed9f6`。不过，为了方便，也允许使用缩写形式；只需指定 GPU UUID 开头足够多的数字，使其能够在目标系统中唯一标识该 GPU 即可。例如，假设系统中没有其他 GPU 共享该前缀，则 `CUDA_VISIBLE_DEVICES=GPU-8932f937` 可以作为上述 GPU UUID 的有效引用。

- [多实例 GPU（Multi-Instance GPU，MIG）](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)支持：`MIG-<GPU-UUID>/<GPU instance ID>/<compute instance ID>`。例如 `MIG-GPU-8932f937-d72c-4106-c12f-20bd9faed9f6/1/2`。只支持枚举单个 MIG 实例。

`cudaGetDeviceCount()` API 返回的设备数量只包含可见设备，因此，使用整数设备标识符的 CUDA API 只支持范围为 \[0, 可见设备数 - 1\] 的序号。GPU 设备的枚举顺序决定序号值。例如，当设置 `CUDA_VISIBLE_DEVICES=2,1` 时，调用 `cudaSetDevice(0)` 会将设备 2 设置为当前设备，因为设备 2 首先被枚举并被分配序号 0。之后调用 `cudaGetDevice(&device_ordinal)` 也会将 `device_ordinal` 设置为 0，该值对应设备 2。

**示例**：

```bash
nvidia-smi -L # Get list of GPU UUIDs
CUDA_VISIBLE_DEVICES=0,1
CUDA_VISIBLE_DEVICES=GPU-8932f937-d72c-4106-c12f-20bd9faed9f6
CUDA_VISIBLE_DEVICES=MIG-GPU-8932f937-d72c-4106-c12f-20bd9faed9f6/1/2
```

<a id="cuda-device-order"></a>

### 5.2.1.2 `CUDA_DEVICE_ORDER`

该环境变量控制 CUDA 枚举可用设备的顺序。

**可能的取值**：

- `FASTEST_FIRST`：使用简单启发式算法，从最快到最慢枚举可用设备（默认值）。

- `PCI_BUS_ID`：按照 PCI 总线 ID 升序枚举可用设备。可以通过 `nvidia-smi --query-gpu=name,pci.bus_id` 获取 PCI 总线 ID。

**示例**：

```bash
CUDA_DEVICE_ORDER=FASTEST_FIRST
CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi --query-gpu=name,pci.bus_id # Get list of PCI bus IDs
```

<a id="cuda-managed-force-device-alloc"></a>

### 5.2.1.3 `CUDA_MANAGED_FORCE_DEVICE_ALLOC`

该环境变量会改变[统一内存](../02-programming-gpus/unified-and-system-memory.html#memory-unified-memory)在多 GPU 系统中的物理存储方式。

**可能的取值**：数值，可以为零或非零。

- **非零值**：强制驱动程序使用设备内存进行物理存储。进程中使用且支持 managed memory 的所有设备，都必须彼此兼容 P2P。否则会返回 `cudaErrorInvalidDevice`。

- `0`：默认行为。

**示例**：

```bash
CUDA_MANAGED_FORCE_DEVICE_ALLOC=0
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 # force device memory
```

<a id="jit-compilation"></a>

## 5.2.2 JIT 编译（JIT Compilation）

<a id="cuda-cache-disable"></a>

### 5.2.2.1 `CUDA_CACHE_DISABLE`

该环境变量控制磁盘上的[即时（Just-In-Time，JIT）编译](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation) Cache 的行为。禁用 JIT Cache 后，每次执行 CUDA 应用时，都会强制执行从 PTX 到 CUBIN 的编译，除非二进制文件中已经包含针对当前运行架构的 CUBIN 代码。

禁用 JIT Cache 会增加应用首次执行时的加载时间。不过，这对于减少应用占用的磁盘空间，以及诊断不同驱动版本或构建标志之间的差异都很有用。

**可能的取值**：

- `1`：禁用 PTX JIT 缓存。

- `0`：启用 PTX JIT 缓存（默认值）。

**示例**：

```bash
CUDA_CACHE_DISABLE=1 # disables caching
CUDA_CACHE_DISABLE=0 # enables caching
```

<a id="cuda-cache-path"></a>

### 5.2.2.2 `CUDA_CACHE_PATH`

该环境变量指定[即时（JIT）编译](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation) Cache 的目录路径。

**可能的取值**：Cache 目录的绝对路径（并且必须具有适当的访问权限）。默认值为：

- Windows：`%APPDATA%\NVIDIA\ComputeCache`

- Linux：`~/.nv/ComputeCache`

**示例**：

```bash
CUDA_CACHE_PATH=~/tmp
```

<a id="cuda-cache-maxsize"></a>

### 5.2.2.3 `CUDA_CACHE_MAXSIZE`

该环境变量以字节为单位指定[即时（JIT）编译](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation) Cache 的大小。超过该大小的二进制文件不会被缓存。如果需要，Cache 会逐出较旧的二进制文件，为较新的二进制文件腾出空间。

**可能的取值**：字节数。默认值为：

- 桌面/服务器平台：`1073741824`（1 GiB）

- 嵌入式平台：`268435456`（256 MiB）

最大大小为 `4294967296`（4 GiB）。

**示例**：

```bash
CUDA_CACHE_MAXSIZE=268435456 # 256 MiB
```

<a id="cuda-force-ptx-jit-and-cuda-force-jit"></a>

### 5.2.2.4 `CUDA_FORCE_PTX_JIT` 与 `CUDA_FORCE_JIT`

这些环境变量指示 CUDA 驱动忽略应用中嵌入的任何 CUBIN，转而对嵌入的 PTX 代码执行[即时（JIT）编译](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation)。

强制执行 JIT 编译会增加应用首次执行时的加载时间。不过，可以用它验证应用中确实嵌入了 PTX 代码，并验证其即时编译是否正常工作，从而确保面向未来架构的[前向兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/)。

`CUDA_FORCE_PTX_JIT` 的优先级高于 `CUDA_FORCE_JIT`。

**可能的取值**：

- `1`：强制执行 PTX JIT 编译。

- `0`：默认行为。

**示例**：

```bash
CUDA_FORCE_PTX_JIT=1
```

<a id="cuda-disable-ptx-jit-and-cuda-disable-jit"></a>

### 5.2.2.5 `CUDA_DISABLE_PTX_JIT` 与 `CUDA_DISABLE_JIT`

这些环境变量会禁用嵌入 PTX 代码的[即时（JIT）编译](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation)，并使用应用中嵌入的兼容 CUBIN。

如果 Kernel 没有嵌入二进制代码，或者嵌入的二进制代码是针对不兼容架构编译的，则 Kernel 将无法加载。这些环境变量可用于验证应用是否为每个 Kernel 都生成了兼容的 CUBIN 代码。更多信息参见[二进制兼容性](../01-introduction/cuda-platform.html#cuda-platform-compute-binary-compatibility)一节。

`CUDA_DISABLE_PTX_JIT` 的优先级高于 `CUDA_DISABLE_JIT`。

**可能的取值**：

- `1`：禁用 PTX JIT 编译。

- `0`：默认行为。

**示例**：

```bash
CUDA_DISABLE_PTX_JIT=1
```

<a id="cuda-force-preload-libraries"></a>

### 5.2.2.6 `CUDA_FORCE_PRELOAD_LIBRARIES`

该环境变量影响为[NVVM](https://docs.nvidia.com/cuda/nvvm-ir-spec/)和[即时（JIT）编译](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation)所需库的预加载。

**可能的取值**：

- `1`：强制驱动在初始化期间预加载 NVVM 和[即时（JIT）编译](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation)所需的库。这会增加内存占用以及 CUDA 驱动初始化所需的时间。涉及多线程的某些死锁场景必须设置该环境变量才能避免。

- `0`：默认行为。

**示例**：

```bash
CUDA_FORCE_PRELOAD_LIBRARIES=1
```

<a id="execution"></a>

## 5.2.3 执行（Execution）

<a id="cuda-launch-blocking"></a>

### 5.2.3.1 `CUDA_LAUNCH_BLOCKING`

该环境变量指定是否禁用或启用异步 Kernel 启动。

禁用异步执行会导致执行速度变慢，但对调试很有用。它会强制 GPU 工作从 CPU 的角度以同步方式运行。这样，CUDA API 错误可以在触发错误的确切 API 调用处被观察到，而不是等到执行后续阶段才发现。同步执行适合调试。

**可能的取值**：

- `1`：禁用异步执行。

- `0`：异步执行（默认值）。

**示例**：

```bash
CUDA_LAUNCH_BLOCKING=1
```

<a id="cuda-device-max-connections"></a>

### 5.2.3.2 `CUDA_DEVICE_MAX_CONNECTIONS`

该环境变量控制并发计算引擎和复制引擎连接（工作队列）的数量，并将两者都设置为指定值。如果独立的 GPU 任务（即从不同 CUDA stream 启动的 Kernel 或复制操作）映射到同一工作队列，就会产生伪依赖。由于这些任务使用相同的底层资源，伪依赖可能导致 GPU 工作串行化。为降低此类伪依赖的概率，建议将此环境变量控制的工作队列数量设置为大于或等于每个 context 中活动 CUDA stream 的数量。

除非通过 `CUDA_DEVICE_MAX_COPY_CONNECTIONS` 环境变量显式设置复制连接数，否则设置该环境变量也会修改复制连接数。

**可能的取值**：`1` 至 `32` 个连接，默认值为 `8`（假定未使用 MPS）。

**示例**：

```bash
CUDA_DEVICE_MAX_CONNECTIONS=16
```

<a id="cuda-device-max-copy-connections"></a>

### 5.2.3.3 `CUDA_DEVICE_MAX_COPY_CONNECTIONS`

该环境变量控制复制操作所涉及的并发复制连接（工作队列）数量。它只影响[计算能力](compute-capabilities.html#compute-capabilities)为 8.0 及更高版本的设备。

如果同时设置了两个变量，`CUDA_DEVICE_MAX_COPY_CONNECTIONS` 会覆盖通过 `CUDA_DEVICE_MAX_CONNECTIONS` 设置的复制连接数。

**可能的取值**：`1` 至 `32` 个连接，默认值为 `8`（假定未使用 MPS）。

**示例**：

```bash
CUDA_DEVICE_MAX_COPY_CONNECTIONS=16
```

<a id="cuda-scale-launch-queues"></a>

### 5.2.3.4 `CUDA_SCALE_LAUNCH_QUEUES`

该环境变量指定可用于启动工作（命令缓冲区）的队列大小缩放因子，也就是可以在设备上排队的待处理 Kernel 或 Host/Device 复制操作总数的缩放因子。

**可能的取值**：`0.25x`、`0.5x`、`2x`、`4x`。

- 除 `0.25x`、`0.5x`、`2x` 或 `4x` 之外的任何值，都会被解释为 `1x`。

**示例**：

```bash
CUDA_SCALE_LAUNCH_QUEUES=2x
```

<a id="cuda-graphs-use-node-priority"></a>

### 5.2.3.5 `CUDA_GRAPHS_USE_NODE_PRIORITY`

该环境变量控制 CUDA Graph 相对于其启动所在 stream 所继承的 stream 优先级的执行优先级。

`CUDA_GRAPHS_USE_NODE_PRIORITY` 会覆盖实例化 Graph 时的[cudaGraphInstantiateFlagUseNodePriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1gd4d586536547040944c05249ee26bc62)标志。

**可能的取值**：

- `0`：继承 Graph 启动所在 stream 的优先级（默认值）。

- `1`：遵循每个节点的启动优先级。CUDA Runtime 将节点级优先级视为对可运行 Graph 节点的调度提示。

**示例**：

```bash
CUDA_GRAPHS_USE_NODE_PRIORITY=1
```

<a id="cuda-device-waits-on-exception"></a>

### 5.2.3.6 `CUDA_DEVICE_WAITS_ON_EXCEPTION`

该环境变量控制 CUDA 应用发生异常（错误）时的行为。

启用后，当设备端异常发生时，CUDA 应用会暂停并等待，从而可以附加[cuda-gdb](https://docs.nvidia.com/cuda/cuda-gdb/index.html)等调试器来检查实时 GPU 状态，之后进程可以退出或继续运行。

**可能的取值**：

- `0`：默认行为。

- `1`：设备异常发生时暂停。

**示例**：

```bash
CUDA_DEVICE_WAITS_ON_EXCEPTION=1
```

<a id="cuda-device-default-persisting-l2-cache-percentage-limit"></a>

### 5.2.3.7 `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT`

该环境变量控制 GPU L2 Cache 中默认“预留（set-aside）”部分的大小。该部分用于[持久化访问](../04-cuda-features/l2-cache-control.html#l2-set-aside)，以 L2 大小的百分比表示。

它适用于支持持久化 L2 Cache 的 GPU，具体来说，是在使用[CUDA 多进程服务（MPS）](https://docs.nvidia.com/deploy/mps/index.html)时计算能力为 8.0 或更高版本的设备。必须在启动 CUDA MPS 控制守护进程之前设置该环境变量，也就是在运行 `nvidia-cuda-mps-control -d` 命令之前设置。

**可能的取值**：0 到 100 之间的百分比值，默认值为 0。

**示例**：

```bash
CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT=25 # 25%
```

<a id="cuda-disable-perf-boost"></a>

### 5.2.3.8 `CUDA_DISABLE_PERF_BOOST`

在 Linux Host 上，将该环境变量设置为 1 会阻止提升设备性能状态；此时可以根据各种启发式规则隐式选择 pstate。该选项可能用于降低功耗，但由于动态选择性能状态，在某些场景下可能导致更高延迟。

**示例**：

```bash
CUDA_DISABLE_PERF_BOOST=1 # perf boost disabled, Linux only.
CUDA_DISABLE_PERF_BOOST=0 # default behavior
```

<a id="cuda-auto-boost-deprecated"></a>

### 5.2.3.9 `CUDA_AUTO_BOOST`（已弃用，deprecated）

该环境变量影响 GPU 时钟的“自动提升（auto boost）”行为，即动态时钟提升。它会覆盖[nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html)工具的“自动提升”选项，即 `nvidia-smi --auto-boost-default=0`。

::: warning
该环境变量已弃用。强烈建议使用 `nvidia-smi --applications-clocks=<memory,graphics>` 或[NVML API](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceCommands.html#group__nvmlDeviceCommands)，而不要使用 `CUDA_AUTO_BOOST` 环境变量。
:::

<a id="module-loading"></a>

## 5.2.4 模块加载（Module Loading）

<a id="cuda-module-loading"></a>

### 5.2.4.1 `CUDA_MODULE_LOADING`

该环境变量影响 CUDA Runtime 加载模块的方式，具体来说，影响它初始化设备代码的方式。

**可能的取值**：

- `DEFAULT`：默认行为，等同于 `LAZY`。

- `LAZY`：延迟加载特定 Kernel，直到通过 `cuModuleGetFunction()` 或 `cuKernelGetFunction()` API 调用提取 CUDA 函数句柄 `CUfunc`。此时，当 CUBIN 中的第一个 Kernel 被加载或访问 CUBIN 中的第一个变量时，才会加载 CUBIN 中的数据。

  - 驱动程序在首次调用 Kernel 时加载所需代码；后续调用不会产生额外开销。这会减少启动时间和 GPU 内存占用。

- `EAGER`：在程序初始化时完全加载 CUDA 模块和 Kernel。CUBIN、FATBIN 或 PTX 文件中的所有 Kernel 和数据，都会在对应的 `cuModuleLoad*` 和 `cuLibraryLoad*` Driver API 调用时完全加载。

  - 启动时间和 GPU 内存占用更高，但 Kernel 启动开销是可预测的。

**示例**：

```bash
CUDA_MODULE_LOADING=EAGER
CUDA_MODULE_LOADING=LAZY
```

<a id="cuda-module-data-loading"></a>

### 5.2.4.2 `CUDA_MODULE_DATA_LOADING`

该环境变量影响 CUDA Runtime 加载与模块关联的数据的方式。

这是对 `CUDA_MODULE_LOADING` 中面向 Kernel 的设置的补充。该环境变量不会影响 Kernel 的 `LAZY` 或 `EAGER` 加载。如果未设置该环境变量，则数据加载行为继承自 `CUDA_MODULE_LOADING`。

**可能的取值**：

- `DEFAULT`：默认行为，等同于 `LAZY`。

- `LAZY`：延迟加载模块数据，直到需要 CUDA 函数句柄 `CUfunc`。此时，当 CUBIN 中的第一个 Kernel 被加载或访问 CUBIN 中的第一个变量时，才会加载 CUBIN 中的数据。

  - 惰性数据加载可能需要 context 同步，从而降低并发执行速度。

- `EAGER`：CUBIN、FATBIN 或 PTX 文件中的所有数据，都会在对应的 `cuModuleLoad*` 和 `cuLibraryLoad*` API 调用时完全加载。

**示例**：

```bash
CUDA_MODULE_DATA_LOADING=EAGER
```

<a id="cuda-binary-loader-thread-count"></a>

### 5.2.4.3 `CUDA_BINARY_LOADER_THREAD_COUNT`

该变量设置加载设备二进制文件时使用的 CPU 线程数。当设置为 0 时，使用的 CPU 线程数将设置为默认值 1。

**可能的取值**：

- 使用的线程数（整数）。默认值为 0，此时使用 1 个线程。

**示例**：

```bash
CUDA_BINARY_LOADER_THREAD_COUNT=4
```

<a id="cuda-error-log-management"></a>

## 5.2.5 CUDA 错误日志管理（CUDA Error Log Management）

<a id="cuda-log-file"></a>

### 5.2.5.1 `CUDA_LOG_FILE`

该环境变量指定一个位置，用于在支持的 CUDA API 调用返回错误时，输出随错误产生的描述性错误日志消息。

例如，如果尝试使用无效的网格配置启动 Kernel，例如 `kernel<<<1, dim3(1,1,128)>>>(...)`，该 Kernel 将启动失败，而 `cudaGetLastError()` 将返回通用的 `invalid configuration argument` 错误。

如果设置了 `CUDA_LOG_FILE` 环境变量，用户可以在日志中看到以下描述性错误消息，从而很容易确定指定的 block z 维度无效：

`[CUDA][E] Block Dimensions (1,1,128) include one or more values that exceed the device limit of (1024,1024,64)`

更多信息参见[错误日志管理](../04-cuda-features/error-log-management.html#error-log-management)。

**可能的取值**：`stdout`、`stderr` 或有效的文件路径（并且必须具有适当的访问权限）。

**示例**：

```bash
CUDA_LOG_FILE=stdout
CUDA_LOG_FILE=/tmp/dbg_cuda_log
```
