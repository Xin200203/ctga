# [M1] 完成 Layer-1 最小闭环验证

## 背景

Layer-1 是 CTGA 的第一层核心贡献，负责把当前帧的 masks、primitives 和 history tracks 组装成 evidence graph，并修复 over-seg / under-seg。

## 目标

- 验证 evidence graph builder、edge scorer、signed graph assembler、partition solver 和 object builder 的连通性
- 在真实样本上导出 graph1 debug packet

## 输出

- 一条从输入帧到 `CurrentObjectHypothesis[]` 的可执行链路
- 对应的可视化与调试材料

## 验收标准

- graph1 构建与求解可完成一次远程 smoke run
- 能导出 primitive clusters 与 current objects
- 可开始评估 fragmentation / merge error

## 建议 Labels

- `method`
- `implementation`
- `priority:high`
