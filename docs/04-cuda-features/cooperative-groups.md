---
title: 4.4 Cooperative Groups
description: Cooperative Groups 线程协作和同步
---

# 4.4 Cooperative Groups

Cooperative Groups 用组对象表达 warp、block、tile、cluster 等协作范围，让同步作用域与算法结构对应。它不能把 block barrier 变成任意 grid 的全局同步；组的参与者和生命周期仍必须满足 API 约束。
