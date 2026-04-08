# CTGA

CTGA 是一个面向在线 RGB-D 场景的 class-agnostic object-centric mapping 研究项目。

当前仓库先按“研究优先、实现渐进”的方式搭建骨架，目标是把方法讨论、实现规划、理论分析、实验设计和代码结构同时整理清楚，避免后续信息散落。

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
