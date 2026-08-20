---
title: 4.9 Asynchronous Barriers
description: 异步 barrier 的到达与等待
---

# 4.9 Asynchronous Barriers

异步 barrier 将“到达”“等待”“推进阶段”分开，适合把数据搬运和计算组织成流水线。参与者数量、到达次数和等待顺序必须一致；否则会产生永久等待或读取未就绪数据。
