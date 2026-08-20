---
title: 3.4 Programming Systems with Multiple GPUs
description: 多 GPU 的设备选择、数据划分与 P2P 通信
---

# 3.4 Programming Systems with Multiple GPUs

多 GPU 系统首先要确定：数据由哪张 GPU 持有，哪些 kernel 在哪张 GPU 执行，设备之间如何复制或通信。PCIe、NVLink、P2P、设备显存和拓扑都会影响设计。

## 设备与工作划分

启动时枚举设备属性，选择 compute capability、显存、SM 数量和互连条件满足要求的设备。不要把设备编号直接当业务语义；枚举顺序可能变化。按数据切分时要处理边界交换和负载均衡；按请求切分时要管理每张 GPU 的队列、context 和内存池。

## P2P 与同步

支持 peer access 时可以减少经过 host memory 的往返，但可用性依赖设备对、拓扑、driver 和系统配置。跨 GPU copy 仍然是异步工作，应使用 stream/event 表达依赖，并用 profiler 确认链路是否成为瓶颈。
