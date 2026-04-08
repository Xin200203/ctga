# CTGA

> Cross-modal Two-Graph Association for online RGB-D class-agnostic object-centric mapping.

CTGA 是一个面向在线 RGB-D 场景的研究型仓库，核心目标是把对象级在线建图中的实例关联问题拆成两个连续图问题：

- 同帧 `evidence graph` 用来修复当前观测的 over-seg / under-seg
- 时序 `association graph` 用来完成当前对象到历史 track 的稳定匹配

仓库当前处于“研究骨架已建立、模块实现持续推进”的阶段，适合作为后续实验、远程训练和 GitHub 协作的主仓库。

## Project Status

- `Status`: active research scaffold
- `Focus`: online RGB-D, class-agnostic object-centric mapping
- `Core idea`: layer-1 observation repair + layer-2 graph matching
- `Current state`: repository structure, docs, core pipelines, training/inference scaffolds, and smoke tests are in place

当前仓库先按“研究优先、实现渐进”的方式搭建骨架，目标是把方法讨论、实现规划、理论分析、实验设计和代码结构同时整理清楚，避免后续信息散落。

## Why This Repository

这个仓库不是在复现单一 baseline，也不是只做一个更强的 decoder。它更偏向一个问题分解明确的研究底座：

- 用 `Layer-1` 验证“同帧跨模态图是否能修复实例碎片化”
- 用 `Layer-2` 验证“二阶关系图匹配是否能降低 ID switch”
- 用结构化文档和模块边界保证研究讨论、工程实现、实验设计保持一致

## 目录概览

- `docs/`: 项目文档主入口，按方法、实现规划、理论分析、实验设计、项目管理分组。
- `src/ctga/`: 主代码包，按数据、地图、primitive、两层图、memory、训练与推理拆分。
- `configs/`: 训练、推理、数据集配置。
- `tools/`: 预处理、回放、可视化、调试脚本入口。
- `tests/`: 单元测试与集成测试。
- `third_party/`: 外部依赖仓库预留目录，如 ESAM、OnlineAnySeg、GASP。
- `data/`: 数据集软链接或本地数据占位目录。
- `cache/`: mask、primitive、label、debug 等缓存输出目录。

## 当前阶段

本仓库按“先建稳定骨架，再逐模块补功能”的方式推进。

建议从以下入口继续推进：

1. 阅读 [docs/README.md](/Users/xin/Code/research/ctga/docs/README.md) 了解文档地图。
2. 阅读 [docs/项目管理/实现执行清单.md](/Users/xin/Code/research/ctga/docs/项目管理/实现执行清单.md) 了解当前开发顺序与检查点。
3. 阅读 [docs/实现规划/仓库结构与模块边界.md](/Users/xin/Code/research/ctga/docs/实现规划/仓库结构与模块边界.md) 对齐代码拆分。
4. 阅读 [docs/实验设计/阶段实验与指标.md](/Users/xin/Code/research/ctga/docs/实验设计/阶段实验与指标.md) 明确阶段目标。
5. 阅读 [docs/项目管理/GitHub首发材料.md](/Users/xin/Code/research/ctga/docs/项目管理/GitHub首发材料.md) 获取仓库简介、labels、milestone 和 issue 规划。
