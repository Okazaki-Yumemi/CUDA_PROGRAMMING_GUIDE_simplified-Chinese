---
title: 4.6 Green Contexts（绿色上下文）
description: Green Context 资源分区、创建、启动和验证的完整中文翻译
---

# 4.6 Green Contexts（绿色上下文）

Green Context（GC）是一种轻量级上下文，在创建时就与一组特定 GPU 资源关联。用户可以在创建 Green Context 时对 GPU 资源进行分区，目前可分区的资源包括流式多处理器（SM）和工作队列（WQ），使得面向某个 Green Context 的 GPU 工作只能使用为该上下文配置的 SM 和工作队列。这样可以减少或更好地控制共享资源造成的干扰。一个应用可以拥有多个 Green Context。

使用 Green Context 不需要修改任何 GPU 代码（kernel），只需要少量主机端改动，例如创建 Green Context，以及为该上下文创建 stream。Green Context 可用于多种场景：例如，在没有其他约束的情况下，确保始终有一些 SM 可供延迟敏感 kernel 开始执行；或者在不修改 kernel 的情况下，快速测试使用更少 SM 的影响。

Green Context 最初通过 [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html) 提供。从 CUDA 13.1 开始，CUDA runtime 通过 execution context（EC）抽象暴露上下文。目前，一个 execution context 可以对应 primary context（runtime API 用户一直隐式交互的上下文），也可以对应 Green Context。本节在指代 Green Context 时会交替使用 *execution context* 和 *Green Context* 这两个术语。

随着 runtime 暴露 Green Context，强烈建议直接使用 CUDA runtime API。本节也只使用 CUDA runtime API。

本节安排如下： [4.6.1](#461 动机与使用时机-motivation-when-to-use) 给出动机示例，[4.6.2](#462-green-contexts易用性-green-contexts-ease-of-use) 展示易用性，[4.6.3](#463-green-context设备资源与资源描述符-green-contexts-device-resource-and-resource-descriptor) 介绍设备资源和资源描述符结构体；[4.6.4](#464-green-context创建示例-green-context-creation-example) 解释如何创建 Green Context，[4.6.5](#465-green-context启动工作-green-contexts---launching-work) 解释如何启动面向它的工作，[4.6.6](#466-其他execution-context-api-additional-execution-contexts-apis) 介绍一些额外 API；最后，[4.6.7](#467-green-context示例-green-contexts-example) 给出完整示例。

## 4.6.1 动机与使用时机（Motivation / When to Use）

启动 CUDA kernel 时，用户不能直接控制该 kernel 将在哪些 SM 上执行。只能通过改变 kernel 的启动几何，或改变会影响每个 SM 最大活动 thread block 数量的因素，间接影响这一点。此外，当多个 kernel 在 GPU 上并行执行（例如运行在不同 CUDA stream 上，或作为 CUDA Graph 的一部分）时，它们也可能争用相同的 SM 资源。

然而，有些场景要求始终有 GPU 资源可用，使延迟敏感工作能够尽快开始并完成。Green Context 通过分区 SM 资源提供了实现途径：给定 Green Context 只能使用特定的 SM，即创建时为它配置的 SM。

图 45 展示了一个例子。假设应用中有两个独立 kernel A 和 B，分别运行在两个不同的非阻塞 CUDA stream 上。kernel A 先启动并占用所有可用 SM 资源。稍后启动延迟敏感的 kernel B 时，已经没有可用 SM。因此，kernel B 只能在 kernel A 开始缩减后才能执行，也就是等 kernel A 的 thread block 执行完成。第一张图展示了关键工作 B 被延迟的情况；纵轴表示被占用 SM 的百分比，横轴表示时间。

![图 45：Green Context 的动机](/images/chapter-04/green_contexts_motivation.png)

使用 Green Context，可以对 GPU 的 SM 进行分区：面向 kernel A 的 Green Context A 访问 GPU 的一部分 SM，面向 kernel B 的 Green Context B 访问剩余 SM。在此设置下，无论 kernel A 的启动配置如何，它只能使用为 Green Context A 配置的 SM。因此，启动关键 kernel B 时，只要没有其他资源约束，就能保证有 SM 可供它立即执行。正如图 45 的第二张图所示，即使 kernel A 的持续时间可能增加，延迟敏感工作 B 也不会再因为 SM 不可用而延迟。图中仅为说明，将 Green Context A 配置为 GPU 的 80% SM。

不需要修改 kernel A 和 B 的代码即可实现这种行为，只需确保它们在属于相应 Green Context 的 CUDA stream 上启动。每个 Green Context 可访问的 SM 数量，应由用户在创建 Green Context 时根据具体情况决定。

**工作队列（Work Queues）**

流式多处理器是可以为 Green Context 配置的一类资源，工作队列是另一类资源。可以把工作队列看作一种黑盒资源抽象，它和其他因素一起影响 GPU 工作执行的并发性。如果独立的 GPU 工作任务（例如提交到不同 CUDA stream 的 kernel）映射到同一工作队列，就可能引入这些任务之间的伪依赖，导致串行执行。用户可以通过环境变量 `CUDA_DEVICE_MAX_CONNECTIONS` 影响 GPU 上工作队列数量的上限。

继续前面的例子，假设工作 B 映射到与工作 A 相同的工作队列。此时即使有可用 SM（Green Context 场景），工作 B 仍可能必须等待工作 A 完全完成。与 SM 类似，用户不能直接控制底层实际使用哪些工作队列。但 Green Context 允许用户用预期并发的按 stream 顺序工作负载数量表示希望达到的最大并发度。驱动随后可以把这个值作为提示，尽量避免不同 execution context 的工作使用同一工作队列，从而防止 execution context 之间发生不必要的干扰。

> **注意**
>
> 即使为不同 Green Context 分配了不同的 SM 资源和工作队列，也不保证独立 GPU 工作一定会并发执行。最好把本节技术理解为移除可能阻止并发执行的因素，即降低潜在干扰，而不是保证并发。

**Green Context 与 MIG 或 MPS 的比较**

为完整起见，本节简要将 Green Context 与另外两种资源分区机制进行比较：[MIG（Multi-Instance GPU）](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html) 和 [MPS（Multi-Process Service）](https://docs.nvidia.com/deploy/mps/index.html)。

MIG 会将支持 MIG 的 GPU 静态分区为多个 MIG 实例（“更小的 GPU”）。分区必须在应用启动前完成，不同应用可以使用不同 MIG 实例。对于持续无法充分利用可用 GPU 资源的应用，使用 MIG 可能有益；随着 GPU 变大，这个问题会更加明显。借助 MIG，不同应用可以在不同 MIG 实例上运行，从而提高 GPU 利用率。MIG 对云服务提供商（CSP）也有吸引力，因为它不仅能提高此类应用的 GPU 利用率，还能为运行在不同 MIG 实例上的客户端提供服务质量（QoS）和隔离。更多细节请参见上面的 MIG 文档。

但 MIG 无法解决前面描述的问题：同一应用的其他 GPU 工作占满所有 SM 资源，导致关键工作 B 延迟。运行在单个 MIG 实例上的应用仍可能遇到这个问题。可以将 Green Context 与 MIG 一起使用，此时可分区的 SM 资源就是给定 MIG 实例的资源。

MPS 主要面向不同进程（例如 MPI 程序），允许它们同时在 GPU 上运行而不进行时间片轮转。应用启动前必须先运行 MPS daemon。默认情况下，MPS 客户端会争用其所在 GPU 或 MIG 实例的所有可用 SM。在多个客户端进程的设置下，MPS 可以使用 active thread percentage 选项动态分区 SM 资源，该选项限制 MPS 客户端进程可以使用的 SM 百分比。与 Green Context 不同，MPS 的 active thread percentage 分区发生在进程级别，通常在应用启动前通过环境变量指定。MPS active thread percentage 表示给定客户端应用最多使用 GPU 的 x% SM，设为 N 个 SM；但这 N 个 SM 可以是 GPU 上任意 N 个 SM，且会随时间变化。相比之下，创建时配置了 N 个 SM 的 Green Context 只能使用这特定的 N 个 SM。

从 CUDA 13.1 开始，如果在启动 MPS 控制 daemon 时显式启用，MPS 也支持静态分区。使用静态分区时，用户必须在应用启动时指定 MPS 客户端进程可以使用的静态分区，此时不再适用 active thread percentage 的动态共享。静态分区模式下的 MPS 与 Green Context 的一个关键区别是：MPS 面向不同进程，而 Green Context 也适用于单个进程内部。此外，与 Green Context 不同，静态分区的 MPS 不允许 SM 资源超额订阅。

借助 MPS，还可以通过带 execution affinity 的 Driver API `cuCtxCreate`，对其创建的 CUDA context 进行程序化 SM 资源分区。这允许一个或多个进程中的不同客户端 CUDA context 各自使用不超过指定数量的 SM。与 active thread percentage 分区一样，这些 SM 可以是 GPU 的任意 SM，并且会随时间变化，不同于 Green Context 的情况。即使存在 MPS 静态分区，也可以使用此选项。需要注意，与 MPS context 相比，创建 Green Context 的开销小得多，因为许多底层结构由 primary context 拥有并共享。

## 4.6.2 Green Context 易用性（Green Contexts: Ease of use）

为说明 Green Context 的易用性，假设有下面的代码：它创建两个 CUDA stream，然后调用一个通过 `<<<>>>` 在这些 stream 上启动 kernel 的函数。如前文所述，除了修改 kernel 的启动几何外，用户无法影响这些 kernel 可以使用多少 SM。

```cpp
int gpu_device_index = 0; // GPU 序号
CUDA_CHECK(cudaSetDevice(gpu_device_index));

cudaStream_t strm1, strm2;
CUDA_CHECK(cudaStreamCreateWithFlags(&strm1, cudaStreamNonBlocking));
CUDA_CHECK(cudaStreamCreateWithFlags(&strm2, cudaStreamNonBlocking));

// 无法控制每个 stream 上运行的 kernel 可以使用多少 SM
code_that_launches_kernels_on_streams(strm1, strm2); // 此函数及 kernel 中抽象掉的部分构成代码的大多数

// 未展示清理代码
```

从 CUDA 13.1 开始，可以使用 Green Context 控制给定 kernel 能访问的 SM 数量。下面的代码展示了实现这一点所需的少量改动：不修改 kernel，只增加几行代码，就能控制在不同 stream 上启动的 kernel 可以使用的 SM 资源。

```cpp
int gpu_device_index = 0; // GPU 序号
CUDA_CHECK(cudaSetDevice(gpu_device_index));

/* ------------------ 创建 Green Context 所需代码 --------------------------- */

// 获取所有可用 GPU SM 资源
cudaDevResource initial_GPU_SM_resources {};
CUDA_CHECK(cudaDeviceGetDevResource(gpu_device_index, &initial_GPU_SM_resources, cudaDevResourceTypeSm));

// 分割 SM 资源。此例创建一个含 16 个 SM 的组和一个含 8 个 SM 的组。
// 假设 GPU 至少有 24 个 SM
cudaDevSmResource result[2] {{}, {}};
cudaDevSmResourceGroupParams group_params[2] =  {
        {.smCount=16, .coscheduledSmCount=0, .preferredCoscheduledSmCount=0, .flags=0},
        {.smCount=8,  .coscheduledSmCount=0, .preferredCoscheduledSmCount=0, .flags=0}};
CUDA_CHECK(cudaDevSmResourceSplit(&result[0], 2, &initial_GPU_SM_resources, nullptr, 0, &group_params[0]));

// 为每个资源生成资源描述符
cudaDevResourceDesc_t resource_desc1 {};
cudaDevResourceDesc_t resource_desc2 {};
CUDA_CHECK(cudaDevResourceGenerateDesc(&resource_desc1, &result[0], 1));
CUDA_CHECK(cudaDevResourceGenerateDesc(&resource_desc2, &result[1], 1));

// 创建 Green Context
cudaExecutionContext_t my_green_ctx1 {};
cudaExecutionContext_t my_green_ctx2 {};
CUDA_CHECK(cudaGreenCtxCreate(&my_green_ctx1, resource_desc1, gpu_device_index, 0));
CUDA_CHECK(cudaGreenCtxCreate(&my_green_ctx2, resource_desc2, gpu_device_index, 0));

/* ------------------ 修改的代码 --------------------------- */

// 只需要使用不同的 CUDA API 创建 stream
cudaStream_t strm1, strm2;
CUDA_CHECK(cudaExecutionCtxStreamCreate(&strm1, my_green_ctx1, cudaStreamDefault, 0));
CUDA_CHECK(cudaExecutionCtxStreamCreate(&strm2, my_green_ctx2, cudaStreamDefault, 0));

/* ------------------ 未修改的代码 --------------------------- */

// 不需要修改此函数或 kernel 中的任何代码。
// 提醒：此函数及 kernel 中抽象掉的部分构成代码的大多数
// 现在，运行在 strm1 上的 kernel 最多使用 16 个 SM，运行在 strm2 上的 kernel 最多使用 8 个 SM。
code_that_launches_kernels_on_streams(strm1, strm2);

// 未展示清理代码
```

前例中的多个 execution context API 接受显式的 `cudaExecutionContext_t` 句柄，因此会忽略调用线程当前的 context。在此之前，不使用 Driver API 的 CUDA runtime 用户默认只与通过 `cudaSetDevice()` 隐式设为线程当前 context 的 primary context 交互。转向基于显式 context 的编程后，语义更容易理解；与依赖线程本地状态（TLS）的隐式 context 编程相比，也可能获得额外收益。

以下各节将详细解释前面代码中的所有步骤。

## 4.6.3 Green Context：设备资源与资源描述符（Green Contexts: Device Resource and Resource Descriptor）

Green Context 的核心是绑定到特定 GPU 设备的设备资源（`cudaDevResource`）。资源可以合并并封装到描述符（`cudaDevResourceDesc_t`）中。Green Context 只能访问创建时所使用描述符封装的资源。

当前 `cudaDevResource` 数据结构定义如下：

```cpp
struct {
     enum cudaDevResourceType type;
     union {
         struct cudaDevSmResource sm;
         struct cudaDevWorkqueueConfigResource wqConfig;
         struct cudaDevWorkqueueResource wq;
     };
 };
```

支持的有效资源类型为 `cudaDevResourceTypeSm`、`cudaDevResourceTypeWorkqueueConfig` 和 `cudaDevResourceTypeWorkqueue`；`cudaDevResourceTypeInvalid` 表示无效资源类型。

有效设备资源可以关联到：

- 一组特定的流式多处理器（SM），资源类型为 `cudaDevResourceTypeSm`；
- 特定的工作队列配置，资源类型为 `cudaDevResourceTypeWorkqueueConfig`；或
- 已存在的工作队列资源，资源类型为 `cudaDevResourceTypeWorkqueue`。

可以分别使用 `cudaExecutionCtxGetDevResource` 和 `cudaStreamGetDevResource` API，查询给定 execution context 或 CUDA stream 是否关联了给定类型的 `cudaDevResource`。execution context 可以同时关联不同类型的设备资源（例如 SM 和工作队列），而 stream 只能关联 SM 类型资源。

默认情况下，一个 GPU 设备拥有全部三种设备资源类型：涵盖该 GPU 所有 SM 的 SM 类型资源、涵盖所有可用工作队列的工作队列配置资源，以及对应的工作队列资源。可以通过 `cudaDeviceGetDevResource` API 获取这些资源。

**相关设备资源结构体概览**

不同资源类型的结构体字段可能由用户显式设置，也可能由相关 CUDA API 设置。建议将所有设备资源结构体进行零初始化。

- SM 类型设备资源 `cudaDevSmResource` 的相关字段：
  - `unsigned int smCount`：此资源中可用的 SM 数量。
  - `unsigned int minSmPartitionSize`：划分此资源所需的最小 SM 数量。
  - `unsigned int smCoscheduledAlignment`：保证在同一个 GPU 处理集群上共同调度的资源内 SM 数量；它与 thread block cluster 相关。当 `flags` 为零时，`smCount` 是该值的倍数。
  - `unsigned int flags`：支持的标志为 0（默认）和 `cudaDevSmResourceGroupBackfill`（参见 `cudaDevSmResourceGroup` 标志）。

  上述字段会由用于创建该 SM 类型资源的划分 API（`cudaDevSmResourceSplitByCount` 或 `cudaDevSmResourceSplit`）设置，或者由获取给定 GPU 设备 SM 资源的 `cudaDeviceGetDevResource` 填充。用户不应直接设置这些字段，详情见后文。

- 工作队列配置设备资源 `cudaDevWorkqueueConfigResource` 的相关字段：
  - `int device`：工作队列资源所在的设备。
  - `unsigned int wqConcurrencyLimit`：为避免伪依赖而预期的按 stream 顺序工作负载数量。
  - `enum cudaDevWorkqueueConfigScope sharingScope`：工作队列资源的共享范围。支持 `cudaDevWorkqueueConfigScopeDeviceCtx`（默认）和 `cudaDevWorkqueueConfigScopeGreenCtxBalanced`。默认选项下，所有工作队列资源在所有 context 之间共享；平衡选项下，驱动会尽可能在 Green Context 之间使用不重叠的工作队列资源，并将用户指定的 `wqConcurrencyLimit` 作为提示。

这些字段必须由用户设置。除了由 `cudaDeviceGetDevResource` 填充的工作队列配置资源外，没有类似划分 API 的 CUDA API 可以生成工作队列配置资源。`cudaDeviceGetDevResource` 可以获取给定 GPU 设备的工作队列配置资源。

- 最后，预先存在的工作队列资源 `cudaDevResourceTypeWorkqueue` 没有可由用户设置的字段。与其他资源类型一样，`cudaDevGetDevResource` 可以获取给定 GPU 设备的预先存在的工作队列资源。

## 4.6.4 Green Context 创建示例（Green Context Creation Example）

创建 Green Context 包含四个主要步骤：

- 步骤 1：从初始资源集合开始，例如获取 GPU 的可用资源；
- 步骤 2：将 SM 资源划分为一个或多个分区（使用可用划分 API 之一）；
- 步骤 3：创建资源描述符，必要时合并不同资源；
- 步骤 4：从描述符创建 Green Context，为它配置资源。

创建 Green Context 后，可以创建属于该上下文的 CUDA stream。随后在此 stream 上启动的 GPU 工作（例如通过 `<<< >>>` 启动的 kernel）只能访问该 Green Context 配置的资源。只要用户将属于 Green Context 的 stream 传给库，库也可以轻松利用 Green Context。详情请参见[Green Context：启动工作](#465-green-context启动工作-green-contexts---launching-work)。

### 4.6.4.1 步骤 1：获取可用 GPU 资源（Step 1: Get available GPU resources）

创建 Green Context 的第一步是获取可用设备资源并填充 `cudaDevResource` 结构体。目前有三种起点：设备、execution context 或 CUDA stream。

相关 CUDA runtime API 签名如下：

- **设备**：`cudaError_t cudaDeviceGetDevResource(int device, cudaDevResource* resource, cudaDevResourceType type)`
- **execution context**：`cudaError_t cudaExecutionCtxGetDevResource(cudaExecutionContext_t ctx, cudaDevResource* resource, cudaDevResourceType type)`
- **stream**：`cudaError_t cudaStreamGetDevResource(cudaStream_t hStream, cudaDevResource* resource, cudaDevResourceType type)`

这些 API 都允许使用所有有效的 `cudaDevResourceType`，但 `cudaStreamGetDevResource` 只支持 SM 类型资源。

通常起点是 GPU 设备。下面的代码展示如何获取给定 GPU 设备的可用 SM 资源。`cudaDeviceGetDevResource` 成功后，用户可以查看资源中可用的 SM 数量。

```cpp
int current_device = 0; // 假设设备序号为 0
CUDA_CHECK(cudaSetDevice(current_device));

cudaDevResource initial_SM_resources = {};
CUDA_CHECK(cudaDeviceGetDevResource(current_device /* GPU device */,
                                   &initial_SM_resources /* 要填充的设备资源 */,
                                   cudaDevResourceTypeSm /* resource type */));

std::cout << "Initial SM resources: " << initial_SM_resources.sm.smCount << " SMs" << std::endl; // 可用 SM 数量

// 与划分相关的特殊字段（见下面的步骤 3）
std::cout << "Min. SM partition size: " << initial_SM_resources.sm.minSmPartitionSize << " SMs" << std::endl;
std::cout << "SM co-scheduled alignment: " << initial_SM_resources.sm.smCoscheduledAlignment << " SMs" << std::endl;
```

也可以获取可用的工作队列配置资源：

```cpp
int current_device = 0; // 假设设备序号为 0
CUDA_CHECK(cudaSetDevice(current_device));

cudaDevResource initial_WQ_config_resources = {};
CUDA_CHECK(cudaDeviceGetDevResource(current_device /* GPU device */,
                                   &initial_WQ_config_resources /* 要填充的设备资源 */,
                                   cudaDevResourceTypeWorkqueueConfig /* resource type */));

std::cout << "Initial WQ config. resources: " << std::endl;
std::cout << "  - WQ concurrency limit: " << initial_WQ_config_resources.wqConfig.wqConcurrencyLimit << std::endl;
std::cout << "  - WQ sharing scope: " << initial_WQ_config_resources.wqConfig.sharingScope << std::endl;
```

`cudaDeviceGetDevResource` 成功后，可以查看该资源的 `wqConcurrencyLimit`。当起点是 GPU 设备时，`wqConcurrencyLimit` 将与环境变量 `CUDA_DEVICE_MAX_CONNECTIONS` 的值或其默认值一致。

### 4.6.4.2 步骤 2：划分 SM 资源（Step 2: Partition SM resources）

创建 Green Context 的第二步，是把可用 `cudaDevResource` SM 资源静态划分为一个或多个分区，也可以将部分 SM 留在剩余分区中。可以使用 `cudaDevSmResourceSplitByCount()` 或 `cudaDevSmResourceSplit()` API。`cudaDevSmResourceSplitByCount()` 只能创建一个或多个*同构*分区以及可能的*剩余*分区；`cudaDevSmResourceSplit()` 还可以创建*异构*分区以及可能的剩余分区。两个 API 都只适用于 SM 类型设备资源。

**`cudaDevSmResourceSplitByCount` API**

`cudaDevSmResourceSplitByCount` runtime API 签名为：

```cpp
cudaError_t cudaDevSmResourceSplitByCount(cudaDevResource* result,
    unsigned int* nbGroups, const cudaDevResource* input,
    cudaDevResource* remaining, unsigned int useFlags,
    unsigned int minCount)
```

用户请求把 `input` SM 类型设备资源划分为 `*nbGroups` 个同构组，每组 `minCount` 个 SM。但最终结果可能包含更新后的 `*nbGroups` 个同构组，每组 `N` 个 SM。更新后的 `*nbGroups` 小于或等于原请求数量，`N` 大于或等于 `minCount`。这些调整可能由架构相关的粒度和对齐要求导致。

![图 46：使用 cudaDevSmResourceSplitByCount API 划分 SM 资源](/images/chapter-04/green_contexts_resource_split_by_count.png)

对于默认的 `useFlags=0` 情况，[计算能力表](../05-technical-appendices/compute-capabilities.html) 列出了当前支持的计算能力的最小 SM 分区大小和 SM 共同调度对齐值。也可以通过 `cudaDevSmResource` 的 `minSmPartitionSize` 和 `smCoscheduledAlignment` 字段取得这些值。某些要求可以通过不同的 `useFlags` 值降低。下表给出请求值与最终结果的相关示例。表格聚焦计算能力 9.0：此时每个分区最少 8 个 SM，且当 `useFlags` 为零时，SM 数量必须是 8 的倍数。

| 请求的 `*nbGroups` | `minCount` | `useFlags` | GH200（132 SM）实际结果 | 剩余 SM | 原因 |
|---:|---:|---|---|---:|---|
| 2 | 72 | 0 | 1 个 72-SM 组 | 60 | 不能超过 132 个 SM |
| 6 | 11 | 0 | 6 个 16-SM 组 | 36 | 必须是 8 的倍数 |
| 6 | 11 | `CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING` | 6 个 12-SM 组 | 60 | 降低为 2 的倍数要求 |
| 2 | 1 | 0 | 2 个 8-SM 组 | 116 | 最少需要 8 个 SM |

表 14：划分功能（Split functionality）。

下面的代码请求将可用 SM 资源划分为 5 个、每个 8 个 SM 的组：

```cpp
cudaDevResource avail_resources = {};
// 填充 avail_resources 的代码未展示

unsigned int min_SM_count = 8;
unsigned int actual_split_groups = 5; // 可能被更新

cudaDevResource actual_split_result[5] = {{}, {}, {}, {}, {}};
cudaDevResource remaining_partition = {};

CUDA_CHECK(cudaDevSmResourceSplitByCount(&actual_split_result[0],
                                         &actual_split_groups,
                                         &avail_resources,
                                         &remaining_partition,
                                         0 /*useFlags */,
                                         min_SM_count));

std::cout << "Split " << avail_resources.sm.smCount << " SMs into " << actual_split_groups << " groups "
          << "with " << actual_split_result[0].sm.smCount << " each "
          << "and a remaining group with " << remaining_partition.sm.smCount << " SMs" << std::endl;
```

需要注意：

- 可以使用 `result=nullptr` 查询将创建的组数；
- 如果不关心剩余分区中的 SM，可以设置 `remaining=nullptr`；
- 剩余分区不具备 `result` 中同构组所具备的相同功能或性能保证；
- 默认情况下 `useFlags` 应为 0，但也支持 `cudaDevSmResourceSplitIgnoreSmCoscheduling` 和 `cudaDevSmResourceSplitMaxPotentialClusterSize`；
- 任何结果 `cudaDevResource` 在先创建资源描述符和 Green Context（即下面的步骤 3 和 4）之前都不能再次划分。

详情请参见 [`cudaDevSmResourceSplitByCount`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION__CONTEXT.html) runtime API 参考。

**`cudaDevSmResourceSplit` API**

前面提到，单次 `cudaDevSmResourceSplitByCount` 调用只能创建同构分区，即每个分区包含相同数量的 SM，外加剩余分区。对于不同 Green Context 上运行的工作有不同 SM 数量需求的异构负载，这可能存在限制。要用按数量划分 API 创建异构分区，通常需要重复步骤 1–4，对已有资源再次划分；或者在步骤 2 中先创建若干个 SM 数量等于所有异构分区 SM 数量最大公约数的同构分区，然后在步骤 3 中合并所需数量的分区。但不建议后一种做法，因为如果一开始请求更大的尺寸，CUDA 驱动可能可以创建更好的分区。

`cudaDevSmResourceSplit` API 通过一次调用创建不重叠的异构分区，解决了上述限制。其 runtime API 签名为：

```cpp
cudaError_t cudaDevSmResourceSplit(cudaDevResource* result,
    unsigned int nbGroups, const cudaDevResource* input,
    cudaDevResource* remainder, unsigned int flags,
    cudaDevSmResourceGroupParams* groupParams)
```

该 API 会根据 `groupParams` 数组中为每个组指定的要求，尝试把 `input` SM 类型资源划分为 `nbGroups` 个有效设备资源（组），放入 `result` 数组。还可以创建可选的剩余分区。划分成功时，如图 47 所示，`result` 中每个资源可以有不同数量的 SM，但不能有零个 SM。

![图 47：使用 cudaDevSmResourceSplit API 划分 SM 资源](/images/chapter-04/green_contexts_resource_split.png)

请求异构划分时，需要为 `result` 中每个资源指定 SM 数量，即对应 `groupParams` 条目的 `smCount` 字段，该值应始终为 2 的倍数。如果应用使用 [Thread Block Clusters](../01-introduction/programming-model.html#programming-model-thread-block-clusters)，仅指定 SM 数量还不够，因为 cluster 中的所有 thread block 保证共同调度。用户还需要通过相关 `groupParams` 条目的 `coscheduledSmCount` 字段，指定该资源组应支持的最大 cluster 尺寸（如果有）。对于计算能力 10.0 及以上的 GPU，cluster 还可以有一个首选尺寸，该尺寸是默认 cluster 尺寸的倍数；在支持的系统上，单次 kernel 启动中会尽可能使用较大的首选尺寸，否则使用较小的默认尺寸。用户可通过 `preferredCoscheduledSmCount` 字段表达该提示。最后，用户可能希望放宽 SM 数量要求，将更多可用 SM 填入给定组；可以把对应 `groupParams` 条目的 `flags` 设为非默认值来启用 backfill。

`cudaDevSmResourceSplit` 还提供**发现模式**，用于事先不知道一个或多个组确切 SM 数量的情况。例如，用户可能希望创建一个在满足共同调度要求（例如允许大小为 4 的 cluster）的同时拥有尽可能多 SM 的设备资源。要使用发现模式，将相关 `groupParams` 条目中的 `smCount` 设为 0。`cudaDevSmResourceSplit` 成功返回后，`groupParams` 的 `smCount` 会填入有效的非零值，即**实际** `smCount`。如果 `result` 非空（不是试运行），相关 `result` 组的 `smCount` 也会设置为同样的值。`nbGroups` 个 `groupParams` 条目的顺序很重要，它们会从左到右、从索引 0 到索引 `nbGroups-1` 依次评估。

表 15 概览了 `cudaDevSmResourceSplit` API 支持的参数：

| `result` | `nbGroups` | `input` | `remainder` | `flags` | `groupParams[i].smCount` | `coscheduledSmCount` | `preferredCoscheduledSmCount` | `groupParams[i].flags` |
|---|---|---|---|---|---|---|---|---|
| 探索性试运行时为 `nullptr`，否则非空 | 组数 | 划分为 `nbGroups` 组的资源 | 不需要剩余组时为 `nullptr` | 0 | 发现模式为 0，否则为有效 `smCount` | 0（默认）或有效的共同调度 SM 数 | 0（默认）或有效的首选共同调度 SM 数（提示） | 0（默认）或 `cudaDevSmResourceGroupBackfill` |

表 15：`cudaDevSmResourceSplit` 划分 API 概览。

注意事项：

1. `cudaDevSmResourceSplit` API 的返回值取决于 `result`：
   - `result != nullptr`：只有划分成功、创建了满足指定要求的 `nbGroups` 个有效 `cudaDevResource` 组时，API 才返回 `cudaSuccess`，否则返回错误。不同错误可能返回同一个错误码，例如 `CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION`；开发阶段建议使用 `CUDA_LOG_FILE` 环境变量获取更详细的错误描述。
   - `result == nullptr`：即使某个组最终的 `smCount` 为零，API 也可能返回 `cudaSuccess`；如果 `result` 非空，同样情况会返回错误。可以把这种模式当作探索支持情况的试运行，尤其适合发现模式。
2. 当 `result != nullptr` 的调用成功时，`result[i]`（`i` 属于 `[0, nbGroups)`）是 `cudaDevResourceTypeSm` 类型，`result[i].sm.smCount` 要么是用户指定的非零 `groupParams[i].smCount`，要么是发现得到的值。在两种情况下，它都必须是 2 的倍数，处于 `[2, input.sm.smCount]` 范围内，并且当 `flags == 0` 时是实际 `groupParams[i].coscheduledSmCount` 的倍数；否则至少等于该值。
3. 将 `coscheduledSmCount` 和 `preferredCoscheduledSmCount` 设置为 0，表示使用当前架构的默认值；这两个默认值都等于通过 `cudaDeviceGetDevResource` 为给定设备获取的 SM 资源（而不是任意 SM 资源）的 `smCoscheduledAlignment`。若要查看默认值，可在二者初始为 0 的成功调用后检查更新的 `groupParams` 条目。
4. 如果存在剩余组，它的 SM 数量和共同调度要求不受上述约束，用户需要自行探索。

下面的代码展示了调用形式：

```cpp
int nbGroups = 2; // 按需更新
unsigned int default_split_flags = 0;
cudaDevResource remainder {}; // 按需更新
cudaDevResource result_use_case[2] = {{}, {}};
cudaDevSmResourceGroupParams group_params_use_case[2] = {
    {.smCount = X, .coscheduledSmCount=0, .preferredCoscheduledSmCount = 0, .flags = 0},
    {.smCount = Y, .coscheduledSmCount=0, .preferredCoscheduledSmCount = 0, .flags = 0}};
CUDA_CHECK(cudaDevSmResourceSplit(&result_use_case[0], nbGroups,
    &initial_GPU_SM_resources, &remainder, default_split_flags,
    &group_params_use_case[0]));
```

表 16 给出一些使用场景下 `cudaDevSmResourceSplit` 参数的示例。表中每一行的 `i` 表示 `groupParams[i]` 条目；条目按升序排列。

| 编号 | 目标/使用场景 | `nbGroups` | `remainder` | `smCount` | `coscheduledSmCount` | `preferredCoscheduledSmCount` | `flags` | `i` |
|---|---|---:|---|---:|---:|---:|---|---:|
| 1 | 一个含 16 个 SM 的资源；不关心剩余 SM；可能使用 cluster。 | 1 | `nullptr` | 16 | 0 | 0 | 0 | 0 |
| 2a | 一个含 16 个 SM 的资源，以及包含其余全部 SM 的资源；不使用 cluster。此行使用剩余组作为第二个资源。 | 1 | 非 `nullptr` | 16 | 2 | 2 | 0 | 0 |
| 2b | 同上，但将第二个资源放入 `result_use_case[1]`。 | 2 | `nullptr` | 16 | 2 | 2 | 0 | 0 |
| 2b | 同上，第二个资源使用发现模式和 backfill。 | 2 | `nullptr` | 0 | 2 | 2 | `cudaDevSmResourceGroupBackfill` | 1 |
| 3 | 两个资源，分别含 28 和 32 个 SM；使用大小为 4 的 cluster。 | 2 | `nullptr` | 28 | 4 | 4 | 0 | 0 |
| 3 | 同上，第二个资源。 | 2 | `nullptr` | 32 | 4 | 4 | 0 | 1 |
| 4 | 一个尽可能多 SM、可运行大小为 8 的 cluster 的资源，以及一个剩余组。 | 1 | 非 `nullptr` | 0 | 8 | 8 | 0 | 0 |
| 5 | 一个尽可能多 SM、可运行大小为 4 的 cluster 的资源，以及一个含 8 个 SM 的资源；注意 `groupParams` 顺序很重要。 | 2 | `nullptr` | 8 | 2 | 2 | 0 | 0 |
| 5 | 同上，第 2 个资源使用发现模式。 | 2 | `nullptr` | 0 | 4 | 4 | 0 | 1 |

表 16：划分 API 的使用场景。

**`cudaDevSmResourceGroupParams` 结构体字段的详细信息**

- `smCount`：控制 `result` 中相应组的 SM 数量。
  - 取值：0（发现模式）或有效非零值（非发现模式）。
  - 有效非零值必须是 2 的倍数，位于 `[2, input->sm.smCount]` 内，并且当 `flags == 0` 时是实际 `coscheduledSmCount` 的倍数，否则大于或等于 `coscheduledSmCount`。
  - 用途：SM 数量未知或不固定时使用发现模式探索可能性；需要指定 SM 数量时使用非发现模式。
  - 注意：发现模式中，非空 `result` 的成功划分调用返回后，实际 SM 数量也满足有效非零值要求。
- `coscheduledSmCount`：控制共同分组（共同调度）的 SM 数量，从而允许在计算能力 9.0 及以上启动不同 cluster；它会影响生成组的 SM 数量和可支持的 cluster 尺寸。
  - 取值：0（当前架构默认值）或有效非零值；有效非零值必须是 2 的倍数且不超过最大限制。
  - 用途：对于 cluster，使用默认值或手动选择值，同时注意给定架构支持的最大可移植 cluster 尺寸。如果代码不使用 cluster，可以使用支持的最小值 2 或默认值。
  - 注意：使用默认值时，成功划分后的实际 `coscheduledSmCount` 也满足有效非零值要求。若 `flags` 非零，结果 `smCount >= coscheduledSmCount`。可以把它理解为有效结果组的底层结构保证：最坏情况下，该组至少能运行一个指定尺寸的 cluster；该结构保证不适用于剩余组，剩余组可启动的 cluster 尺寸需要用户自行探索。
- `preferredCoscheduledSmCount`：向驱动提供提示，尽量在可能时把实际 `coscheduledSmCount` 个 SM 的组，合并成 `preferredCoscheduledSmCount` 个 SM 的更大组。这可以让代码使用计算能力 10.0 及以上设备的首选 cluster 尺寸功能。
  - 取值：0（当前架构默认值）或有效非零值；有效非零值必须是实际 `coscheduledSmCount` 的倍数。
  - 用途：如果使用首选 cluster 且设备为计算能力 10.0（Blackwell）或更高，选择大于 2 的手动值。如果不使用 cluster，选择与 `coscheduledSmCount` 相同的值：二者都选择支持的最小值 2，或者二者都使用 0。
  - 注意：使用默认值时，成功划分后的实际 `preferredCoscheduledSmCount` 也满足有效非零值要求。
- `flags`：控制结果组的 SM 数量是否为实际共同调度 SM 数量的倍数（默认），或者是否允许把 SM backfill 到该组。在 backfill 情况下，结果 `result[i].sm.smCount` 大于或等于指定的 `groupParams[i].smCount`。
  - 取值：0（默认）或 `cudaDevSmResourceGroupBackfill`。
  - 用途：默认使用 0，使结果组保证支持多个指定 `coscheduledSmCount` 尺寸的 cluster；如果希望组中尽可能多地获得 SM，可使用 backfill，但 backfill 的 SM 不提供共同调度保证。
  - 注意：使用 backfill 标志创建的组仍可能支持 cluster，例如保证至少支持一个 `coscheduledSmCount` 尺寸。

### 4.6.4.3 步骤 2（续）：添加工作队列资源（Step 2 (continued): Add workqueue resources）

如果还希望指定工作队列资源，则必须显式完成。下面的示例为指定设备创建工作队列配置资源，使用平衡共享范围和 4 的并发限制。

```cpp
cudaDevResource split_result[2] = {{}, {}};
// 填充 split_result[0] 的代码未展示；使用 nbGroups=1 的划分 API

// 最后一个资源是工作队列资源。
split_result[1].type = cudaDevResourceTypeWorkqueueConfig;
split_result[1].wqConfig.device = 0; // 假设设备序号为 0
split_result[1].wqConfig.sharingScope = cudaDevWorkqueueConfigScopeGreenCtxBalanced;
split_result[1].wqConfig.wqConcurrencyLimit = 4;
```

工作队列并发限制 4 向驱动提示用户预期最多 4 个并发的按 stream 顺序工作负载。驱动会在可能的情况下尝试遵守该提示分配工作队列。

### 4.6.4.4 步骤 3：创建资源描述符（Step 3: Create a Resource Descriptor）

资源划分后，下一步是使用 `cudaDevResourceGenerateDesc` API，为预期由 Green Context 使用的所有资源生成资源描述符。

```cpp
cudaError_t cudaDevResourceGenerateDesc(cudaDevResourceDesc_t *phDesc,
    cudaDevResource *resources, unsigned int nbResources)
```

可以合并多个 `cudaDevResource` 资源。例如，下面的代码将三个资源组封装在一个资源描述符中；只需保证这些资源在 `resources` 数组中连续。

```cpp
cudaDevResource actual_split_result[5] = {};
// 填充 actual_split_result 的代码未展示

// 生成资源描述符，封装 actual_split_result[2] 到 [4] 共 3 个资源
cudaDevResourceDesc_t resource_desc;
CUDA_CHECK(cudaDevResourceGenerateDesc(&resource_desc, &actual_split_result[2], 3));
```

也支持合并不同类型的资源，例如同时包含 SM 和工作队列资源的描述符。

要使 `cudaDevResourceGenerateDesc` 调用成功：

- 全部 `nbResources` 个资源必须属于同一个 GPU 设备；
- 如果合并多个 SM 类型资源，它们应来自同一个划分 API 调用，并具有相同的 `coscheduledSmCount` 值（如果不属于剩余分区）；
- 最多只能存在一个工作队列配置资源或工作队列类型资源。

### 4.6.4.5 步骤 4：创建 Green Context（Step 4: Create a Green Context）

最后一步是使用 `cudaGreenCtxCreate` API 从资源描述符创建 Green Context。该 Green Context 只能访问创建时指定的资源描述符所封装的资源（例如 SM、工作队列），这些资源会在此步骤中配置。

```cpp
cudaError_t cudaGreenCtxCreate(cudaExecutionContext_t *phCtx,
    cudaDevResourceDesc_t desc, int device, unsigned int flags)
```

`flags` 参数应设为 0。还建议在创建 Green Context 前，使用 `cudaInitDevice` 或 `cudaSetDevice` API 显式初始化设备的 primary context；后者还会将 primary context 设为调用线程的当前 context。这样可以确保创建 Green Context 时不会再产生额外的 primary context 初始化开销。

```cpp
int current_device = 0; // 假设只有一个 GPU
CUDA_CHECK(cudaSetDevice(current_device)); // 或 cudaInitDevice

cudaDevResourceDesc_t resource_desc {};
// 生成 resource_desc 的代码未展示

// 在 current_device 指定的 GPU 上创建 green_ctx，允许访问 resource_desc 中的资源
cudaExecutionContext_t green_ctx {};
CUDA_CHECK(cudaGreenCtxCreate(&green_ctx, resource_desc, current_device, 0));
```

Green Context 创建成功后，用户可以针对每种资源类型，在该 execution context 上调用 `cudaExecutionCtxGetDevResource` 验证资源。

**创建多个 Green Context**

一个应用可以拥有多个 Green Context，此时需要重复上面的部分步骤。多数场景中，每个 Green Context 会获得一组独立且不重叠的已配置 SM。例如，对于包含 5 个同构 `cudaDevResource` 组的 `actual_split_result` 数组，一个 Green Context 的描述符可以封装 `actual_split_result[2]` 到 `[4]`，另一个 Green Context 的描述符封装 `[0]` 到 `[1]`；此时某个特定 SM 只会配置给两个 Green Context 中的一个。

也可以进行 SM 超额订阅，并在某些场景中使用。例如，可以允许第二个 Green Context 的描述符封装 `actual_split_result[0]` 到 `[2]`。这样，`actual_split_result[2]` 中的所有 SM 都会超额订阅，即同时配置给两个 Green Context；而 `[0]` 到 `[1]` 以及 `[3]` 到 `[4]` 中的资源可能只被两个 Green Context 中的一个使用。应根据具体情况谨慎使用 SM 超额订阅。

## 4.6.5 Green Context：启动工作（Green Contexts - Launching work）

要启动面向前述步骤创建的 Green Context 的 kernel，首先需要使用 `cudaExecutionCtxStreamCreate` API 为该 Green Context 创建 stream。在该 stream 上使用 `<<< >>>` 或 `cudaLaunchKernel` API 启动 kernel，可以确保 kernel 只能使用 execution context 为该 stream 提供的资源（SM、工作队列）。例如：

```cpp
// 为之前创建的 green_ctx Green Context 创建 green_ctx_stream CUDA stream
cudaStream_t green_ctx_stream;
int priority = 0;
CUDA_CHECK(cudaExecutionCtxStreamCreate(&green_ctx_stream,
                                        green_ctx,
                                        cudaStreamDefault,
                                        priority));

// my_kernel 只能使用 green_ctx_stream 的 execution context 可用的资源
my_kernel<<<grid_dim, block_dim, 0, green_ctx_stream>>>();
CUDA_CHECK(cudaGetLastError());
```

传给上述 stream 创建 API 的默认 stream 创建标志，在 `green_ctx` 为 Green Context 时等价于 `cudaStreamNonBlocking`。

**CUDA Graphs**

对于作为 CUDA Graph 一部分启动的 kernel，还有一些细节需要注意。与 kernel 不同，启动 CUDA Graph 所用的 CUDA stream 不决定使用哪些 SM 资源，因为该 stream 只用于依赖跟踪。

kernel 节点（以及其他适用节点类型）将在哪个 execution context 上执行，在节点创建时确定。如果使用 stream capture 创建 CUDA Graph，则参与 capture 的 stream 的 execution context 决定相关图节点的 execution context。如果使用 Graph API 创建，则用户应显式设置每个相关节点的 execution context。例如，添加 kernel 节点时，应使用类型为 `cudaGraphNodeTypeKernel` 的多态 `cudaGraphAddNode` API，并显式设置 `cudaKernelNodeParamsV2` 结构体 `.kernel` 下的 `.ctx` 字段。`cudaGraphAddKernelNode` 不允许用户指定 execution context，因此应避免使用。一个 Graph 中不同节点可以属于不同 execution context。

验证时，可以在节点跟踪模式下使用 Nsight Systems（`--cuda-graph-trace node`），观察特定 Green Context 的图节点在哪个上下文执行。注意，在默认的 *graph* 跟踪模式中，整个 Graph 会显示在启动它的 stream 所属 Green Context 下；但如前所述，这不能提供各图节点 execution context 的信息。

从程序上验证时，可以使用 CUDA Driver API `cuGraphKernelNodeGetParams(graph_node, &node_params)`，将 `node_params.ctx` 上下文句柄与该图节点的预期上下文句柄比较。由于 `CUgraphNode` 和 `cudaGraphNode_t` 可以互换使用，因此可以调用 Driver API；但用户需要包含相关 `cuda.h` 头文件并直接链接驱动（`-lcuda`）。

**Thread Block Clusters**

带 thread block cluster 的 kernel 和其他 kernel 一样，也可以在 Green Context stream 上启动，因此使用该 Green Context 配置的资源。创建设备资源时，前面的[步骤 2](#4642-步骤-2划分-sm-资源-step-2-partition-sm-resources) 展示了如何指定需要共同调度的 SM 数量，以支持 cluster。但与任何使用 cluster 的 kernel 一样，用户应使用相关 occupancy API 确定 kernel 的最大潜在 cluster 尺寸（通过 `cudaOccupancyMaxPotentialClusterSize`），必要时确定最大活动 cluster 数量（通过 `cudaOccupancyMaxActiveClusters`）。如果将 Green Context stream 指定为相关 `cudaLaunchConfig` 的 `stream` 字段，这些 occupancy API 会考虑该 Green Context 配置的 SM 资源。这对于可能接收用户传入 Green Context CUDA stream 的库，以及 Green Context 从剩余设备资源创建的场景尤其有用。

```cpp
// 假设 cudaStream_t gc_stream 已创建，且存在 __global__ void cluster_kernel。

// 如果可能，取消注释以支持非可移植 cluster 尺寸
// CUDA_CHECK(cudaFuncSetAttribute(cluster_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1))

cudaLaunchConfig_t config = {0};
config.gridDim          = grid_dim; // 必须是 cluster 尺寸的倍数
config.blockDim         = block_dim;
config.dynamicSmemBytes = expected_dynamic_shared_mem;

cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeClusterDimension;
attribute[0].val.clusterDim.x = 1;
attribute[0].val.clusterDim.y = 1;
attribute[0].val.clusterDim.z = 1;
config.attrs = attribute;
config.numAttrs = 1;

config.stream=gc_stream; // 传入将用于该 kernel 的 CUDA stream

int max_potential_cluster_size = 0;
// 下一次调用会忽略启动配置中的 cluster 尺寸
CUDA_CHECK(cudaOccupancyMaxPotentialClusterSize(&max_potential_cluster_size, cluster_kernel, &config));
std::cout << "max potential cluster size is " << max_potential_cluster_size << " for CUDA stream gc_stream" << std::endl;

// 可以选择用 max_potential_cluster_size 更新启动配置的 clusterDim。
// 这样会使相同 kernel 和启动配置的 cudaLaunchKernelEx 调用成功。

int num_clusters= 0;
CUDA_CHECK(cudaOccupancyMaxActiveClusters(&num_clusters, cluster_kernel, &config));
std::cout << "Potential max. active clusters count is " << num_clusters << std::endl;
```

**验证 Green Context 的使用**

除了观察 Green Context 配置对相关 kernel 执行时间的经验影响外，用户还可以借助 [Nsight Systems](https://developer.nvidia.com/nsight-systems) 或 [Nsight Compute](https://developer.nvidia.com/nsight-compute) CUDA 开发工具，在一定程度上验证 Green Context 是否被正确使用。

例如，在属于不同 Green Context 的 CUDA stream 上启动的 kernel，会在 Nsight Systems 报告 CUDA HW timeline 部分的不同 Green Context 行中显示。Nsight Compute 在 Session 页面提供 Green Context Resources 概览，并在 Details 部分的 Launch Statistics 中显示更新后的 SM 数量。前者提供配置资源的可视化位掩码。当应用使用不同 Green Context 时，这尤其有用：用户可以确认不同 GC 之间的预期重叠（不重叠，或 SM 超额订阅时预期的非零重叠）。

图 48 展示了一个示例：两个 Green Context 分别配置了 112 和 16 个 SM，二者之间没有 SM 重叠。该视图可以帮助用户验证每个 Green Context 的配置 SM 数量，也能确认没有 SM 超额订阅，因为没有任何方格在两个 Green Context 中都标记为绿色。

![图 48：Nsight Compute 中的 Green Context 资源](/images/chapter-04/green_contexts_ncu_mask.png)

Launch Statistics 部分还会明确列出为该 Green Context 配置的 SM 数量，因此可以用它表示该 kernel 执行期间能够访问的 SM 数量。需要注意，这表示 kernel 可以访问的 SM，而不是 kernel 实际运行所在的 SM 数量。前面的资源概览同样如此。kernel 实际使用的 SM 数量取决于多种因素，包括 kernel 本身（启动几何等）以及 GPU 上同时运行的其他工作。

## 4.6.6 其他 Execution Context API（Additional Execution Contexts APIs）

本节介绍一些额外的 Green Context API。完整列表请参见相关的 [CUDA runtime API 章节](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION__CONTEXT.html)。

对于使用 CUDA event 的同步，可以使用 `cudaError_t cudaExecutionCtxRecordEvent(cudaExecutionContext_t ctx, cudaEvent_t event)` 和 `cudaError_t cudaExecutionCtxWaitEvent(cudaExecutionContext_t ctx, cudaEvent_t event)` API。`cudaExecutionCtxRecordEvent` 会记录一个 CUDA event，在调用时捕获指定 execution context 的所有工作/活动；`cudaExecutionCtxWaitEvent` 会使提交到该 execution context 的所有未来工作等待指定 event 中捕获的工作。

如果 execution context 有多个 CUDA stream，使用 `cudaExecutionCtxRecordEvent` 比 `cudaEventRecord` 更方便。若不用该 execution context API，要达到等价行为，就需要在该 execution context 的每个 stream 上分别调用 `cudaEventRecord` 记录 event，然后让依赖工作分别等待所有 event。同样，如果需要 execution context 的所有 stream 等待 event 完成，`cudaExecutionCtxWaitEvent` 比 `cudaStreamWaitEvent` 更方便；替代方案是对该 execution context 中的每个 stream 分别调用 `cudaStreamWaitEvent`。

在 CPU 侧执行阻塞同步，可以使用 `cudaError_t cudaExecutionCtxSynchronize(cudaExecutionContext_t ctx)`。该调用会阻塞，直到指定 execution context 完成全部工作。如果指定的 execution context 不是通过 `cudaGreenCtxCreate` 创建，而是通过 `cudaDeviceGetExecutionCtx` 获取、因而是设备的 primary context，那么调用该函数还会同步在同一设备上创建的所有 Green Context。

可以使用 `cudaExecutionCtxGetDevice` 获取给定 execution context 关联的设备，使用 `cudaExecutionCtxGetId` 获取给定 execution context 的唯一标识符。

最后，可以通过 `cudaError_t cudaExecutionCtxDestroy(cudaExecutionContext_t ctx)` API 销毁显式创建的 execution context。

## 4.6.7 Green Context 示例（Green Contexts Example）

本节展示 Green Context 如何让关键工作更早开始并完成。与[4.6.1](#461-动机与使用时机-motivation-when-to-use)中的场景类似，应用有两个 kernel，分别运行在两个不同的非阻塞 CUDA stream 上。CPU 侧的时间线如下：首先在 CUDA stream `strm1` 上启动一个长时间运行的 kernel（`delay_kernel_us`），它需要在完整 GPU 上运行多个 wave。短暂等待（小于该 kernel 的持续时间）后，在 `strm2` 上启动一个较短但关键的 kernel（`critical_kernel`）。测量两个 kernel 的 GPU 执行时间，以及从 CPU 启动到完成的时间。

作为长时间运行 kernel 的替代，使用一个 delay kernel：每个 thread block 固定运行若干微秒，并且 thread block 数量超过 GPU 可用 SM 数量。

开始时不使用 Green Context，但将关键 kernel 启动在优先级高于长 kernel 的 CUDA stream 上。由于 stream 优先级较高，关键 kernel 可以在长 kernel 的一些 thread block 完成后立即开始执行。然而，它仍需等待某些可能运行很久的 thread block 完成，因而启动会延迟。

图 49 展示了 Nsight Systems 报告中的这个场景。长 kernel 在 stream 13 上启动，短而关键的 kernel 在 stream 14 上启动，stream 14 具有更高优先级。如图中所示，关键 kernel 在本例中等待了 0.9ms 才能开始执行。如果两个 stream 的优先级相同，关键 kernel 会晚得多。

![图 49：不使用 Green Context 的 Nsight Systems 时间线](/images/chapter-04/green_contexts_nsys_example_no_GCs_with_prio.png)

为了利用 Green Context，创建两个 Green Context，每个都配置一组不同且不重叠的 SM。以 132 个 SM 的 H100 为例，为说明目的选择将 16 个 SM 分配给关键 kernel（Green Context 3），将 112 个 SM 分配给长 kernel（Green Context 2）。如图 50 所示，现在关键 kernel 几乎可以立即开始，因为只有 Green Context 3 能使用专门为它保留的 SM。

与单独运行相比，短 kernel 的持续时间可能增加，因为它能使用的 SM 数量受到限制。长 kernel 也一样：它不能再使用 GPU 的全部 SM，而会受到其 Green Context 配置资源的限制。然而，关键结果是关键 kernel 工作现在可以比之前明显更早开始并完成。当然这还要排除其他限制，因为如前文所述，并行执行并不能保证。

![图 50：使用 Green Context 的 Nsight Systems 时间线](/images/chapter-04/green_contexts_nsys_example_w_GCs.png)

在所有情况下，都应在实验后根据具体情况决定确切的 SM 划分。
