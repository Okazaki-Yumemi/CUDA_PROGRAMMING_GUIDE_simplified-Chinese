---
title: 5.5 Floating-Point Computation
description: CUDA 浮点格式、精度、舍入与 FMA
---

# 5.5 Floating-Point Computation

本节涵盖浮点格式、normal/subnormal、特殊值、舍入、FMA、精度、IEEE-754、CUDA 数学函数及 fast math。host/device 结果出现 bit-level 差异时，应检查运算顺序、融合、编译选项、架构和数据类型，而不是简单判定一侧错误。
