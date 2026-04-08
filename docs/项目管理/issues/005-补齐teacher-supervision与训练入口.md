# [M1] 补齐 teacher supervision 与训练入口

## 背景

当前 supervision 和 trainer 已经有骨架，但需要在真实训练前进一步对齐 teacher replay、标签生成和 batch 输入规范。

## 目标

- 补齐 teacher replay 缓存与标签构建约定
- 固化 stage-1 / stage-2 训练输入输出
- 明确训练脚本的远程运行方式

## 输出

- 训练数据准备说明
- 训练入口示例
- 相关文档回填

## 验收标准

- stage-1 / stage-2 至少能完成 dry-run
- 标签结构与 loss 调用保持一致

## 建议 Labels

- `implementation`
- `evaluation`
- `priority:medium`
