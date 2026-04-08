# CTGA Checklist

这份清单是仓库级入口，详细说明见 [docs/项目管理/开发清单.md](/Users/xin/Code/research/ctga/docs/项目管理/开发清单.md) 与 [docs/项目管理/实现执行清单.md](/Users/xin/Code/research/ctga/docs/项目管理/实现执行清单.md)。

## 基础模块

- [x] `common`
- [x] `datasets`
- [x] `frontends`
- [x] `mapping`
- [x] `primitives`

## Layer-1

- [x] evidence graph builder
- [x] edge features
- [x] edge scorer
- [x] signed graph assembler
- [x] partition solver
- [x] current object builder

## Layer-2

- [x] association graph builder
- [x] unary features / scorer
- [x] relation features / scorer
- [x] candidate gating
- [x] component builder
- [x] Hungarian fallback
- [x] beam-QAP solver
- [x] QP-relax placeholder

## Memory / Supervision / Training

- [x] track bank
- [x] lifecycle / feature fusion
- [x] primitive GT assigner
- [x] edge label builder
- [x] teacher track replay
- [x] association label builder
- [x] loss functions
- [x] stage-1 / stage-2 trainer entry
- [x] joint-train placeholder

## Inference / Tools / Tests

- [x] online engine
- [x] evaluator / diagnostics
- [x] replay / visualize / debug tools
- [x] smoke tests
- [x] 文档归档与导航整理

## 远程运行前建议

- 在远程服务器或工作站安装 `pyproject.toml` 中声明的依赖
- 先跑 `pytest`
- 再跑 sequence replay 或 online inference 的 smoke case
