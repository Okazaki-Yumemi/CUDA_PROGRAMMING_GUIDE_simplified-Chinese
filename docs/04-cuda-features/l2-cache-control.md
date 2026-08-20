---
title: 4.13 L2 Cache 控制
description: 使用 CUDA API 控制持久化访问的 L2 Cache
---

# 4.13 L2 Cache 控制（L2 Cache Control）

当 CUDA kernel 重复访问全局内存中的某个数据区域时，这类访问可以视为持久化访问（persisting）。如果数据只访问一次，则可以视为流式访问（streaming）。

计算能力 8.0 及更高版本的设备能够影响数据在 L2 cache 中的持久化状态，从而可能以更高带宽和更低延迟访问全局内存。

该功能通过两个主要 API 暴露：

- CUDA Runtime API（从 CUDA 11.0 开始）提供对 L2 cache 持久化的程序化控制。
- libcu++ 库中的 `cuda::annotated_ptr` API（从 CUDA 11.5 开始）为 CUDA kernel 中的指针标注内存访问属性，以实现类似效果。详见 [libcu++ 文档](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_access_properties/annotated_ptr.html)。

下面各节聚焦 CUDA Runtime API。关于 `cuda::annotated_ptr` 方法的详细信息，请参阅 [libcu++ 文档](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_access_properties/annotated_ptr.html)。

## 4.13.1 为持久化访问预留 L2 Cache（L2 Cache Set-Aside for Persisting Accesses）

可以从 L2 cache 中预留一部分，用于全局内存的持久化数据访问。持久化访问会优先使用这部分预留空间；普通或流式全局内存访问只能在该空间未被持久化访问使用时利用它。

持久化访问的 L2 cache 预留大小可以在限制范围内调整：

```cpp
cudaGetDeviceProperties(&prop, device_id);
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* 为持久化访问预留 3/4 的 L2 cache，或预留允许的最大值 */
```

当 GPU 配置为多实例 GPU（MIG）模式时，L2 cache 预留功能被禁用。

使用多进程服务（MPS）时，不能通过 `cudaDeviceSetLimit` 更改 L2 cache 预留大小。此时只能在启动 MPS server 时，通过环境变量 `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT` 指定预留大小。

## 4.13.2 持久化访问的 L2 策略（L2 Policy for Persisting Accesses）

访问策略窗口（access policy window）指定一段连续的全局内存区域，以及该区域内访问在 L2 cache 中的持久化属性。

下面的代码示例展示如何使用 CUDA Stream 设置 L2 持久化访问窗口。

**CUDA Stream 示例**

```cpp
cudaStreamAttrValue stream_attribute;                                         // Stream 级属性数据结构
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // 全局内存数据指针
stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // 持久化访问的字节数
                                                                              // （必须小于 cudaDeviceProp::accessPolicyMaxWindowSize）
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Cache 命中率提示
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // 命中时的访问属性类型
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // 未命中时的访问属性类型

// 为 cudaStream_t 类型的 CUDA stream 设置属性
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
```

当 CUDA `stream` 随后执行 kernel 时，全局内存范围 `[ptr..ptr+num_bytes)` 内的访问，比其他全局内存位置的访问更可能在 L2 cache 中保持持久化。

也可以为 CUDA Graph 的 Kernel Node 设置 L2 持久化，如下面的示例所示。

**CUDA GraphKernelNode 示例**

```cpp
cudaKernelNodeAttrValue node_attribute;                                     // Kernel 级属性数据结构
node_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // 全局内存数据指针
node_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // 持久化访问的字节数
                                                                            // （必须小于 cudaDeviceProp::accessPolicyMaxWindowSize）
node_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Cache 命中率提示
node_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // 命中时的访问属性类型
node_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // 未命中时的访问属性类型

// 为 cudaGraphNode_t 类型的 CUDA Graph Kernel 节点设置属性
cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute);
```

`hitRatio` 参数可以指定获得 `hitProp` 属性的访问比例。在上面两个示例中，全局内存区域 `[ptr..ptr+num_bytes)` 中 60% 的内存访问具有持久化属性，40% 的访问具有流式属性。哪些具体访问被归类为持久化访问（即获得 `hitProp`）是随机的，其概率约为 `hitRatio`；概率分布取决于硬件架构和内存范围。

例如，假设 L2 预留 cache 大小为 16 KB，`accessPolicyWindow` 中的 `num_bytes` 为 32 KB：

- 当 `hitRatio` 为 0.5 时，硬件会随机选择 32 KB 窗口中的 16 KB，将其标记为持久化并缓存到 L2 cache 预留区域。
- 当 `hitRatio` 为 1.0 时，硬件会尝试将整个 32 KB 窗口缓存到 L2 cache 预留区域。由于预留区域小于窗口，cache line 会被驱逐，以便让 32 KB 数据中最近使用的 16 KB 保留在 L2 cache 预留部分。

因此，可以使用 `hitRatio` 避免 cache line 抖动，并总体减少移入和移出 L2 cache 的数据量。

小于 1.0 的 `hitRatio` 值还可以手动控制并发 CUDA stream 中不同 `accessPolicyWindow` 能够在 L2 中缓存的数据量。例如，假设 L2 预留 cache 大小为 16 KB；两个不同 CUDA stream 中的并发 kernel 各有一个 16 KB 的 `accessPolicyWindow`，且二者的 `hitRatio` 都为 1.0。在争用共享 L2 资源时，它们可能驱逐对方的 cache line。但是，如果两个 `accessPolicyWindow` 的 `hitRatio` 都为 0.5，则它们不太可能驱逐自身或对方的持久化 cache line。

## 4.13.3 L2 访问属性（L2 Access Properties）

不同的全局内存数据访问定义了三种访问属性：

1. `cudaAccessPropertyStreaming`：具有流式属性的内存访问不太可能在 L2 cache 中持久化，因为这类访问会被优先驱逐。
2. `cudaAccessPropertyPersisting`：具有持久化属性的内存访问更可能在 L2 cache 中持久化，因为这类访问会优先保留在 L2 cache 的预留部分。
3. `cudaAccessPropertyNormal`：该访问属性会强制将之前应用的持久化访问属性重置为普通状态。来自之前 CUDA kernel、具有持久化属性的内存访问，可能在预期用途结束很久之后仍保留在 L2 cache 中。这种使用后的持久化会减少后续不使用持久化属性的 kernel 可用的 L2 cache。使用 `cudaAccessPropertyNormal` 属性重置访问策略窗口，会移除之前访问的持久化（优先保留）状态，效果如同之前访问没有使用访问属性。

## 4.13.4 L2 持久化示例（L2 Persistence Example）

下面的示例展示如何为持久化访问预留 L2 cache、通过 CUDA Stream 在 CUDA kernel 中使用预留的 L2 cache，然后重置 L2 cache。

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);                                                                  // 创建 CUDA stream

cudaDeviceProp prop;                                                                        // CUDA 设备属性变量
cudaGetDeviceProperties( &prop, device_id);                                                 // 查询 GPU 属性
size_t size = min( int(prop.l2CacheSize * 0.75) , prop.persistingL2CacheMaxSize );
cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size);                                  // 为持久化访问预留 3/4 的 L2，或允许的最大值

size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes);                        // 选择用户指定 num_bytes 与最大窗口大小中的较小值

cudaStreamAttrValue stream_attribute;                                                       // Stream 级属性数据结构
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data1);             // 全局内存数据指针
stream_attribute.accessPolicyWindow.num_bytes = window_size;                                // 持久化访问的字节数
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                                        // Cache 命中率提示
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;               // 持久化属性
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;                // 未命中时的访问属性

cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // 为 CUDA Stream 设置属性

for(int i = 0; i < 10; i++) {
    cuda_kernelA<<<grid_size,block_size,0,stream>>>(data1);                                 // kernel 多次使用 data1
}                                                                                           // [data1 + num_bytes) 受益于 L2 持久化
cuda_kernelB<<<grid_size,block_size,0,stream>>>(data1);                                     // 同一 stream 中的另一个 kernel 也可受益
                                                                                            // 于 data1 的持久化

stream_attribute.accessPolicyWindow.num_bytes = 0;                                          // 将窗口大小设为 0 以禁用窗口
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // 覆盖 CUDA Stream 的访问策略属性
cudaCtxResetPersistingL2Cache();                                                            // 移除 L2 中所有持久化 line

cuda_kernelC<<<grid_size,block_size,0,stream>>>(data2);                                     // data2 现在可以在普通模式下使用完整 L2
```

## 4.13.5 将 L2 访问重置为普通状态（Reset L2 Access to Normal）

之前 CUDA kernel 的持久化 L2 cache line 可能在使用结束很久之后仍保留在 L2 中。因此，对于流式或普通内存访问而言，将 L2 cache 重置为普通状态很重要，这样它们就能以普通优先级使用 L2 cache。持久化访问有三种方式可以重置为普通状态：

- 使用访问属性 `cudaAccessPropertyNormal` 重置之前的持久化内存区域。
- 调用 `cudaCtxResetPersistingL2Cache()`，将所有持久化 L2 cache line 重置为普通状态。
- 最终，长期未触碰的 line 会自动重置为普通状态。不建议依赖自动重置，因为自动重置发生所需的时间长度是不确定的。

## 4.13.6 管理 L2 预留 Cache 的利用率（Manage Utilization of L2 set-aside cache）

在不同 CUDA stream 中并发执行的多个 CUDA kernel，可能为各自的 stream 分配了不同的访问策略窗口。但是，L2 预留 cache 部分由所有这些并发 CUDA kernel 共享。因此，该预留部分的总体利用量是所有并发 kernel 各自使用量的总和。当持久化访问量超过 L2 预留 cache 容量时，将内存访问标记为持久化的收益会降低。

要管理 L2 预留 cache 部分的利用率，应用必须考虑以下因素：

- L2 预留 cache 的大小。
- 可能并发执行的 CUDA kernel。
- 所有可能并发执行的 CUDA kernel 的访问策略窗口。
- 何时以及如何执行 L2 重置，使普通或流式访问能够以相同优先级使用之前预留的 L2 cache。

## 4.13.7 查询 L2 Cache 属性（Query L2 cache Properties）

与 L2 cache 相关的属性属于 `cudaDeviceProp` 结构体，可以使用 CUDA Runtime API `cudaGetDeviceProperties` 查询。

CUDA 设备属性包括：

- `l2CacheSize`：GPU 上可用 L2 cache 的大小。
- `persistingL2CacheMaxSize`：可以为持久化内存访问预留的最大 L2 cache 大小。
- `accessPolicyMaxWindowSize`：访问策略窗口的最大大小。

## 4.13.8 控制持久化内存访问的 L2 Cache 预留大小（Control L2 Cache Set-Aside Size for Persisting Memory Access）

持久化内存访问的 L2 预留 cache 大小使用 CUDA Runtime API `cudaDeviceGetLimit` 查询，使用 CUDA Runtime API `cudaDeviceSetLimit` 以 `cudaLimit` 的形式设置。该限制可设置的最大值是 `cudaDeviceProp::persistingL2CacheMaxSize`。

```cpp
enum cudaLimit {
    /* 省略其他字段 */
    cudaLimitPersistingL2CacheSize
};
```
