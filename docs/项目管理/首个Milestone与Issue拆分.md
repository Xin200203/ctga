# 首个 Milestone 与 Issue 拆分

## Milestone

名称：

`M1: Baseline-Ready Core Pipeline`

目标：

- 固化研究仓库的第一轮工程主线
- 让远程服务器上的 baseline、回放、调试、最小训练都能开始工作
- 把后续方法迭代建立在稳定的输入、图构建与评测接口上

## Issue 组织原则

- 每个 issue 尽量只覆盖一个可验收目标
- 每个 issue 都要明确输入、输出和验收条件
- 优先按“基础依赖 -> graph1 -> graph2 -> training/eval”顺序推进

## 建议顺序

1. 第三方依赖与 baseline 对齐
2. 数据契约与 sequence loader 补全
3. layer-1 最小闭环验证
4. layer-2 最小闭环验证
5. teacher supervision 与训练入口
6. 远程运行、评测与 baseline 固化

## 里程碑完成条件

- baseline 依赖目录准备完成
- 远程环境可安装并导入仓库
- 两个 smoke tests 通过
- sequence replay 能产出 debug packet
- 第一轮 baseline 对比脚本具备运行入口
