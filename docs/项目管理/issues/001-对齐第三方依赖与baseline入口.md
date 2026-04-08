# [M1] 对齐第三方依赖与 baseline 入口

## 背景

当前仓库已经建立了主代码骨架，但 `third_party/` 仍然只是占位目录。为了让后续 baseline 对比和参考实现落地，需要先把 ESAM、OnlineAnySeg、GASP 的接入边界明确下来。

## 目标

- 明确 `third_party/ESAM`、`third_party/OnlineAnySeg`、`third_party/GASP` 的接入方式
- 在文档中记录各自承担的职责
- 预留统一的 baseline / reference 入口

## 输出

- 第三方仓库接入说明
- baseline 入口文档
- 必要时补充 wrapper 或路径配置文件

## 验收标准

- 文档明确说明三个 third-party 仓库分别服务于哪个模块
- 后续开发者不需要再猜“该改主仓库还是 third_party”

## 建议 Labels

- `implementation`
- `docs`
- `priority:high`
