# [M1] 完成 Layer-2 最小闭环验证

## 背景

Layer-2 是 CTGA 的第二层核心贡献，负责将当前对象图与历史 track 图做 candidate-gated graph matching。

## 目标

- 验证 association graph builder、candidate gating、component builder、Hungarian fallback、beam-QAP solver 的闭环
- 在重复物体案例上输出 association 调试信息

## 输出

- 一条从 `CurrentObjectHypothesis[]` 到 track assignment 的可执行链路
- unary / pairwise compatibility 可视化或调试包

## 验收标准

- graph2 能在远程环境跑通最小样例
- 可输出 candidate map、component 划分和最终匹配
- 可开始比较 Hungarian vs beam-QAP

## 建议 Labels

- `method`
- `implementation`
- `priority:high`
