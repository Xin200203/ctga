# Configs

本目录存放训练、推理、数据集与消融实验配置。

当前建议的配置层次：

- `base.yaml`: 全局默认项
- `dataset_*.yaml`: 数据集相关路径与预处理参数
- `train_layer1.yaml`: 第一层 evidence graph 训练配置
- `train_layer2.yaml`: 第二层 association graph 训练配置
- `infer_online.yaml`: 在线推理配置
