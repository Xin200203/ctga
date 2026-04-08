# GitHub 首发材料

## Repository Description

推荐放在 GitHub 仓库简介里的短描述：

`Cross-modal two-graph association for online RGB-D class-agnostic object-centric mapping.`

如果你想保留一点研究属性，也可以用这版：

`Research scaffold for layer-1 evidence graph and layer-2 graph matching in online RGB-D object-centric mapping.`

## README 顶部信息目标

README 首页应该让第一次进仓库的人立刻知道四件事：

1. 这是一个什么研究问题
2. 核心方法和已有工作差异在哪里
3. 仓库当前做到什么程度
4. 下一步应该去看哪些文档

这些内容已经同步体现在 [README.md](/Users/xin/Code/research/ctga/README.md) 顶部。

## Suggested GitHub Topics

建议添加这些 topics：

- `rgb-d`
- `3d-vision`
- `object-centric-mapping`
- `instance-segmentation`
- `graph-matching`
- `scene-understanding`
- `research`
- `pytorch`

## Suggested Labels

建议先建一组简单但够用的 labels：

- `research`
- `method`
- `implementation`
- `dataset`
- `evaluation`
- `docs`
- `good first issue`
- `blocked`
- `priority:high`
- `priority:medium`
- `priority:low`

## First Milestone

里程碑名称：

`M1: Baseline-Ready Core Pipeline`

里程碑目标：

- 跑通可复现的数据输入与缓存读取
- 跑通 layer-1 evidence graph 的最小闭环
- 跑通 layer-2 association graph 的最小闭环
- 具备在线回放、调试导出、基础 smoke test 能力

完成标准：

- `tests/test_geometry.py` 与 `tests/test_graph_pipeline.py` 能在远程环境通过
- 能执行一次最小 sequence replay
- 关键模块接口不再频繁变动
- 第一轮 baseline 对比可以正式启动

## First Issue Set

建议首批 issue 不要太散，围绕同一个 milestone 推进。已经准备好的 issue 草稿在：

- [docs/项目管理/issues/001-对齐第三方依赖与baseline入口.md](/Users/xin/Code/research/ctga/docs/项目管理/issues/001-对齐第三方依赖与baseline入口.md)
- [docs/项目管理/issues/002-补全数据契约与序列读取链路.md](/Users/xin/Code/research/ctga/docs/项目管理/issues/002-补全数据契约与序列读取链路.md)
- [docs/项目管理/issues/003-完成layer1最小闭环验证.md](/Users/xin/Code/research/ctga/docs/项目管理/issues/003-完成layer1最小闭环验证.md)
- [docs/项目管理/issues/004-完成layer2最小闭环验证.md](/Users/xin/Code/research/ctga/docs/项目管理/issues/004-完成layer2最小闭环验证.md)
- [docs/项目管理/issues/005-补齐teacher-supervision与训练入口.md](/Users/xin/Code/research/ctga/docs/项目管理/issues/005-补齐teacher-supervision与训练入口.md)
- [docs/项目管理/issues/006-建立远程运行与评测基线.md](/Users/xin/Code/research/ctga/docs/项目管理/issues/006-建立远程运行与评测基线.md)

## 发布建议顺序

1. 更新 GitHub 仓库简介与 topics
2. 推送当前 README
3. 创建 milestone `M1: Baseline-Ready Core Pipeline`
4. 依次创建上面 6 个 issue，并挂到该 milestone 下
5. 先把 `001~004` 标为 `priority:high`
