---
title: 4.2 CUDA Graphs
description: CUDA Graph 的定义、实例化和执行
---

# 4.2 CUDA Graphs

CUDA Graph 将 kernel、复制、event 和依赖边组成可实例化、可重复提交的执行图。定义、instantiate、update 和 execute 是不同阶段；大量重复的小任务可以减少 host 提交开销，但图更新与资源准备也有成本。

图中没有显式依赖边的节点不应假定有先后关系。Stream capture 适合把已有 stream 工作流记录成图，Graph API 适合直接描述拓扑。
