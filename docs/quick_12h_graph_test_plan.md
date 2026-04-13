# 12h 快速测试工程计划（含可视化与样例挖掘）

## 0. 目标

这版 12h 原型的唯一目标不是追论文最终精度，而是快速回答下面三个问题：

1. **当前帧的 2D mask + 3D primitive + 历史先验**，能否把明显的 over-segmentation 修得更好。
2. 在得到更稳定的当前对象之后，**简单的一阶 current-to-memory matching** 是否已经足够稳定。
3. 当出现错误时，是否能通过中间可视化明确定位问题来自：
   - Layer-1 当前观测修复；
   - Layer-2 历史关联；
   - 或 primitive 构建本身。

---

## 1. 这 12h 版本必须强制收缩的范围

### 必做
- posed RGB-D sequence（已知 pose）
- 单场景 / 单短序列（100~300 帧）
- class-agnostic
- hand-crafted score + 简单 solver
- 强可视化 + 中间结果 dump
- 使用 GT 进行 over-seg sample mining

### 暂时不做
- 不训练网络
- 不做可微 QP / second-order matching solver
- 不做 open-vocabulary semantics
- 不接完整 ESAM 主干
- 不做大规模 benchmark 跑分
- 不做完整 TSDF 高质量重建

### 12h 原型的核心定位
- **不是最终系统**
- 是一个 **diagnostic prototype**：验证机制、生成可视化、定位问题、决定下一步是否值得继续投入。

---

## 2. 12h 版本最终必须交付的产物

### A. 可运行的最小原型
输入：RGB、depth、pose、intrinsic、2D masks、GT（仅用于 sample mining 与 debug）
输出：
- 当前帧 primitives
- Layer-1 clusters
- 当前对象与历史 tracks 的一阶匹配
- 更新后的 track bank

### B. 样例级可视化
每个 sample：连续 8~16 帧，展示 over-seg 现象及修复过程。

必须输出：
- `strip.png`：连续时序条带图
- `video.mp4`：连续短视频
- `score_matrix.png`：对象-轨迹分数热力图
- `meta.json`：样例说明与统计

### C. 总览级可视化
- `success_grid.png`
- `fail_grid.png`
- `merge_error_grid.png`

### D. 诊断信息
- 每帧 debug packet
- GT-based fragmentation / purity / match accuracy
- 简短总结：下一步该补哪一层

---

## 3. 推荐工程目录

```text
quick_graph_test/
  data/
  cache_masks/
  out/
  src/
    io_seq.py
    mask_source.py
    geometry.py
    primitive_build.py
    score_l1.py
    cluster_l1.py
    track_bank.py
    assoc_l2.py
    sample_mining.py
    viz.py
    run_quick_test.py
```

### 每个文件职责
- `io_seq.py`：读取 RGB / depth / pose / intrinsic
- `mask_source.py`：读取缓存 masks；支持 oracle/debug 模式
- `geometry.py`：反投影、法向估计、投影、bbox 计算、voxelize
- `primitive_build.py`：当前帧 over-segmentation
- `score_l1.py`：构建 `MP/PP/PT/(MT)` 边和 signed graph
- `cluster_l1.py`：Layer-1 聚类与当前对象构建
- `track_bank.py`：活跃轨迹查询、更新、新生/休眠
- `assoc_l2.py`：Layer-2 一阶关联 + Hungarian / greedy
- `sample_mining.py`：基于 GT 自动挖掘 over-seg samples
- `viz.py`：所有图像、视频、热力图、gallery 导出
- `run_quick_test.py`：主入口

---

## 4. 输入数据约定

建议统一成下面格式：

```text
scene_root/
  color/
  depth/
  pose/
  intrinsic.txt
  gt/                 # 可选，用于 sample mining / debug
```

### `FramePacket`

```python
@dataclass
class FramePacket:
    frame_id: int
    rgb: np.ndarray           # [H, W, 3], uint8
    depth: np.ndarray         # [H, W], float32, meters
    pose_c2w: np.ndarray      # [4, 4]
    K: np.ndarray             # [3, 3]
    scene_id: str
```

### `Mask2D`

```python
@dataclass
class Mask2D:
    mask_id: int
    bitmap: np.ndarray        # [H, W], bool
    bbox_xyxy: np.ndarray     # [4]
    area: int
    score: float
    feat2d: np.ndarray        # [D2] or None
```

### `Primitive3D`

```python
@dataclass
class Primitive3D:
    prim_id: int
    pixel_idx: np.ndarray     # [Np, 2]
    xyz: np.ndarray           # [Np, 3]
    voxel_ids: np.ndarray     # [Nv]
    center_xyz: np.ndarray    # [3]
    bbox_xyzxyz: np.ndarray   # [6]
    normal_mean: np.ndarray   # [3]
    color_mean: np.ndarray    # [3]
```

### `TrackState`

```python
@dataclass
class TrackState:
    track_id: int
    voxel_ids: np.ndarray
    center_xyz: np.ndarray
    bbox_xyzxyz: np.ndarray
    feat_color_mean: np.ndarray
    last_seen: int
    age: int
    miss_count: int
    confidence: float
    status: str  # active / dormant / dead
```

### `CurrentObject`

```python
@dataclass
class CurrentObject:
    obj_id_local: int
    primitive_ids: list[int]
    support_mask_ids: list[int]
    support_track_ids: list[int]
    voxel_ids: np.ndarray
    center_xyz: np.ndarray
    bbox_xyzxyz: np.ndarray
```

---

## 5. 核心算法：12h 版本如何“缩水”但保持主逻辑

## 5.1 Primitive 构建：先从单帧 RGB-D 做过分割

### 原则
不要先做全局 3D 地图上的 primitive；12h 版先从**单帧图像平面 + 深度**做过分割，再 lift 到 3D。

### 推荐做法
在像素网格上做 4/8 邻域连通分割，连接条件：

- 深度差：`abs(z_i - z_j) < tau_z`
- 法向角：`angle(n_i, n_j) < tau_n`
- 颜色差：`||c_i - c_j|| < tau_c`

### 初始阈值建议
- `tau_z = 0.03 ~ 0.05 m`
- `tau_n = 20° ~ 30°`
- `tau_c = 20 ~ 30`（RGB L2）

### 目标状态
- 一把椅子 / 一张桌子通常被切成 3~15 个 primitive
- 宁可稍微 over-seg，也不要一开始大块粘连

### 输出可视化
- `primitive_overlay_2d.png`
- `primitive_cloud_3d.png`

---

## 5.2 Layer-1：先做当前观测修复

### 节点
- 当前 2D masks：`M_t`
- 当前 3D primitives：`P_t`
- 历史 active tracks：`T_{t-1}^{act}`（只作为 soft support）

### 边
- `MP`：mask–primitive
- `PP`：primitive–primitive
- `PT`：primitive–track
- `MT`：可选，若时间够再加

### 关键原则
- **优化变量只落在 primitive 上**
- 历史 tracks 只作为 evidence / prior，不做最终 assignment
- Layer-1 的输出是：**当前对象划分**

### `MP` 分数（建议）

$$
a_{ij}^{MP} = 0.4\cdot cover + 0.3\cdot contain + 0.2\cdot depth\_cons + 0.1\cdot color\_sim
$$

其中：
- `cover`：primitive 投影像素有多少比例在 mask 内
- `contain`：mask 对 primitive 的非对称包含率
- `depth_cons`：mask 深度与 primitive 投影深度一致性
- `color_sim`：颜色统计一致性

### `PP` 分数（建议）

$$
a_{jj'}^{PP} = 0.5\cdot adj + 0.3\cdot normal\_sim + 0.2\cdot color\_sim
$$

其中：
- `adj`：3D 邻接 / bbox 接触
- `normal_sim`：法向夹角相似
- `color_sim`：颜色相似

### `PT` 分数（建议）

$$
a_{jk}^{PT} = 0.5\cdot vote + 0.3\cdot bbox\_iou - 0.2\cdot center\_dist
$$

其中：
- `vote`：primitive 的 voxel 落入 track 支持区域附近的比例
- `bbox_iou`：primitive bbox 与 track bbox 的相容性
- `center_dist`：中心距离惩罚

### signed primitive graph 组装

$$
W_{jj'}^+ = \lambda_{pp} a_{jj'}^{PP} + \lambda_m \sum_i a_{ij}^{MP} a_{ij'}^{MP} + \lambda_t \sum_k a_{jk}^{PT} a_{j'k}^{PT}
$$

$$
W_{jj'}^- = \lambda_c \cdot track\_conflict(j,j') + \lambda_u \cdot mask\_conflict(j,j')
$$

### 12h 求解器建议
先不要上复杂 multicut：
1. 阈值化强正边
2. union-find 合并
3. 再用简单冲突规则拆边

如果 agent 很快，也可以加 `networkx.connected_components` + 冲突剪枝。

### Layer-1 输出
- primitive cluster ids
- 当前对象 `CurrentObject[]`

---

## 5.3 Layer-2：一阶 current-to-memory matching

### 12h 版目标
只做最小 unary current-to-memory association，不做 second-order。

### 输入
- 当前对象 `O_t`
- 历史 active tracks `T_{t-1}^{act}`

### unary 分数

$$
\theta_{rk} = 0.5\cdot geom + 0.3\cdot app + 0.2\cdot hist
$$

其中：
- `geom`：bbox overlap / center distance / voxel vote
- `app`：当前对象与 track 的简单颜色 / embedding 相似
- `hist`：来自 layer-1 的 support track 累积支持

### candidate gating
每个当前对象只保留：
- 距离足够近的 tracks
- bbox 尺寸相容的 tracks
- score top-k（建议 `k=3~5`）

### 求解
优先使用：
- `scipy.optimize.linear_sum_assignment`（Hungarian）

备选：
- 简单 greedy + 去重

### newborn 规则
若所有候选分数都低于阈值 `tau_newborn`，则该对象作为 newborn。

### track 生命周期
- matched：更新 track
- unmatched current object：newborn
- unmatched track：`miss_count += 1`
- miss 超阈值：dormant / delete

---

## 5.4 GT 的作用（这版一定要利用）

这版数据已经包含 GT，所以 GT 主要用于两件事：

### 1. 自动挖 over-seg sample
不是人工挑例子，而是自动找“连续几帧里某个 GT 实例一直被切碎”的片段。

### 2. 做 debug 诊断
不是最终论文指标，而是帮助你定位错误来源：
- Layer-1 purity
- Layer-1 fragmentation
- Layer-2 match accuracy

### 明确：GT 不应该默认进入主推理路径
主推理仍然是 zero-shot / class-agnostic 风格；
GT 只用于：
- sample mining
- debug mode
- 对照可视化

### 可选 debug mode
可以加一个 `--oracle_masks` 开关：
- 用 GT 2D masks 代替外部 2D masks
- 用来快速判断问题在 2D 前端还是在 graph 逻辑

---

## 6. 过分割样例挖掘（sample mining）

## 6.1 样例定义
一个 sample 是一个连续时间窗口 `[t0, t1]`，其中某个 GT 实例在多个连续帧里被明显过分割。

### 推荐设置
- 窗口长度：12 帧（默认）
- 可选：8 / 16 帧

---

## 6.2 GT-based fragmentation score

对于 GT instance `g` 在帧 `t`：

$$
S_{frag}(g,t) = \#\{p_j \mid p_j \text{ overlaps GT instance } g\}
$$

也就是：
- 某个 GT 实例在当前帧被多少个 primitive 覆盖

### 作为 over-seg 候选的规则
若满足：
- `S_frag(g,t) >= 4`（或更高）
- 连续至少 3~5 帧都高
- GT 在这些帧中可见面积足够大

则该 `(g, [t0,t1])` 作为候选 sample。

---

## 6.3 额外筛选：更值得展示的 sample

优先展示三类：

### A. split fixed
Layer-1 前 primitive 数多，Layer-1 后 cluster 数明显变少，且没有明显误并。

### B. still fragmented
明显 over-seg，但 Layer-1 修复失败。

### C. wrong merge
Layer-1 过度合并，把两个 GT instance 并到一起。

### 推荐导出数量
- 成功修复：10 个
- 修复失败：5 个
- 错误合并：5 个

---

## 7. 可视化总方案（必须与样例挖掘合并）

可视化不是附属功能，而是 12h 原型的核心产物。

---

## 7.1 每个 sample 的主可视化：时序条带图（必须）

### 形式
对一个 sample 的连续 12 帧，横向排成时间条带。

### 每一列固定展示 5 行
1. RGB crop
2. 2D masks overlay
3. 原始 primitives overlay
4. Layer-1 clusters overlay
5. history tracks / final match overlay

### 每列附加数字
- `#prim`
- `#cluster`
- `track_id / newborn`

### 关键要求
- crop 必须用整个时间窗口的 **union bbox + margin**，不能逐帧抖动
- 同一个历史 track 的颜色要跨帧固定

### 输出文件
```text
out/samples/sample_xxx/strip.png
```

---

## 7.2 每个 sample 的视频 / GIF（必须）

### 每帧大面板建议布局
- 左：RGB + 2D masks
- 中：primitives + Layer-1 clusters
- 右：history tracks + final assignment

### 输出文件
```text
out/samples/sample_xxx/video.mp4
```

### 作用
视频用于发现问题；条带图用于汇报展示。

---

## 7.3 全局 + 局部双视图面板（强烈建议）

### 形式
- 左：全局完整图像 / 全局点云
- 右：局部 sample crop 的时序条带图

### 用途
方便解释：这个局部对象在整个场景中的位置和上下文。

### 输出文件
```text
out/samples/sample_xxx/global_local.png
```

---

## 7.4 score matrix 可视化（必须）

### 内容
当前对象 × 历史 track 的 unary score 热力图。

### 输出文件
```text
out/samples/sample_xxx/score_matrix.png
```

### 作用
- 看 top‑1 候选是否合理
- 看二义性是否集中出现在重复物体
- 为后续 second-order 版本预留参照

---

## 7.5 gallery 总览图（强烈建议）

### 推荐生成三个总览图
- `success_grid.png`
- `fail_grid.png`
- `merge_error_grid.png`

### 每个格子内容
- 一帧代表性 RGB crop
- Layer-1 cluster 结果
- 标签：`split fixed` / `still fragmented` / `wrong merge`
- 可选：`#prim -> #cluster`

### 用途
适合向导师快速汇报“系统性观察到了哪些现象”。

---

## 7.6 颜色与标注规范（必须统一）

### primitive
- 同帧随机色即可
- 不要求跨帧一致

### cluster
- 当前 sample 内尽量稳定
- 若最终匹配成功，可继承历史 track 颜色

### track
- 整条序列固定颜色

### 状态色
- 绿色：匹配稳定 / 修复成功
- 黄色：低置信度 / 边界情况
- 红色：明显错误（wrong merge / ID switch / still fragmented）

### 文本标注
建议固定显示：
- `cluster_id`
- `track_id`
- `status`
- `score`

---

## 8. 可视化自动导出的目录结构

```text
out/
  samples/
    sample_000/
      meta.json
      strip.png
      video.mp4
      score_matrix.png
      global_local.png
      frames/
        frame_000123_panel.png
        frame_000124_panel.png
        ...
    sample_001/
    ...
  galleries/
    success_grid.png
    fail_grid.png
    merge_error_grid.png
```

### `meta.json` 建议字段

```json
{
  "scene_id": "scene0000_00",
  "gt_instance_id": 17,
  "frame_start": 123,
  "frame_end": 134,
  "sample_type": "split_fixed",
  "before_avg_primitives": 8.4,
  "after_avg_clusters": 1.9,
  "matched_track_id": 5,
  "notes": "history support helps merge chair back and seat"
}
```

---

## 9. 分阶段执行计划（12 小时）

## 0h – 1h：数据读入与基础 overlay

### 任务
- 读取 RGB / depth / pose / masks / GT
- 输出单帧 RGB、depth、mask overlay

### 验收
- 第一帧能正确显示
- mask 与 RGB 对齐
- depth 尺度正常

---

## 1h – 2.5h：primitive builder

### 任务
- 单帧 RGB-D 过分割
- primitive lift 到 3D
- 输出 primitive 2D / 3D 可视化

### 验收
- 一般物体被切成多个合理碎片
- 不出现全图一个大块或全图噪声点

---

## 2.5h – 4.5h：Layer-1 基础版（MP + PP）

### 任务
- 实现 `MP` 与 `PP` 边
- 构建 signed primitive graph
- union-find / connected components 聚类
- 导出当前 cluster 可视化

### 验收
- cluster 结果比原始 primitive 更接近对象块

---

## 4.5h – 6h：引入历史 support（PT）

### 任务
- 实现最小 track bank
- 实现 `PT`
- 在 layer-1 中使用历史作为 soft prior
- 导出 track support 可视化

### 验收
- 同一历史对象的碎片更容易并起来
- 两个明显不同历史对象的碎片不轻易误并

---

## 6h – 7.5h：Layer-2 一阶关联

### 任务
- 从 layer-1 输出当前对象
- 构建 unary score matrix
- 做 candidate gating + Hungarian/greedy
- 更新 track bank

### 验收
- 大物体在相邻帧中大体保持稳定 ID

---

## 7.5h – 9.5h：sample mining + strip/video 渲染

### 任务
- 基于 GT fragmentation 自动挖 sample
- 导出每个 sample 的 `strip.png` 与 `video.mp4`
- 导出 `score_matrix.png`

### 验收
- 至少得到 10 个有效 sample
- 至少 3 个成功修复案例可直接肉眼观察

---

## 9.5h – 10.5h：GT debug mode

### 任务
- 计算 cluster purity
- 计算 fragmentation 前后变化
- 计算 match accuracy / provisional ID switch

### 验收
- 能够判断错误主要来自 layer-1 还是 layer-2

---

## 10.5h – 12h：整理输出与总结

### 任务
- 导出 gallery
- 生成一条示例视频
- 写 `summary.txt`

### `summary.txt` 至少包含
1. 当前观测修复是否有效
2. 历史 support 是否有帮助
3. 一阶 current-to-memory 是否足够稳定
4. 最明显的失败类型是什么
5. 下一步最该补哪一层

---

## 10. 给 agent 的逐步任务单

不要让 agent 一次写完整系统，按下面顺序逐步完成。

### Task 1
实现 `io_seq.py`、`mask_source.py`，并输出单帧 overlay。

### Task 2
实现 `primitive_build.py`，输出 `primitive_overlay_2d.png` 和 `primitive_cloud_3d.png`。

### Task 3
实现 `score_l1.py` 中的 `MP/PP`，以及 `cluster_l1.py` 的基础聚类。

### Task 4
实现 `track_bank.py` 最小版本，以及 `PT` support。

### Task 5
实现 `assoc_l2.py` 的 unary + Hungarian。

### Task 6
实现 `sample_mining.py`，基于 GT 自动挖 12 帧 over-seg sample。

### Task 7
实现 `viz.py`：
- strip
- video
- score matrix
- gallery

### Task 8
实现 GT debug metrics 和 `summary.txt`。

---

## 11. 每一步的强制验收标准

### 数据层
- [ ] 能输出第一帧 RGB、depth、mask overlay

### primitive 层
- [ ] 2D primitive overlay 正常
- [ ] 3D primitive cloud 正常

### layer-1
- [ ] 当前 cluster 比 primitive 更接近对象
- [ ] 能输出正/负边可视化

### layer-2
- [ ] 至少在短序列中大物体 ID 稳定
- [ ] 能输出对象-轨迹分数矩阵

### sample mining
- [ ] 自动找到连续 over-seg 片段
- [ ] 每个 sample 都有连续 8~16 帧

### visualization
- [ ] 每个 sample 都有 `strip.png`
- [ ] 每个 sample 都有 `video.mp4`
- [ ] 总览图可生成

---

## 12. 这版原型最重要的两个 baseline

### baseline A：不做 layer-1 修复
当前 primitive / mask 直接去匹配历史 tracks。

### baseline B：做 layer-1，但 layer-2 只用 unary Hungarian
这是 12h 版主 baseline。

之后若这两版都跑通，再决定是否值得加 second-order layer-2。

---

## 13. 成功标准

只要满足下面 5 条，这个 12h 原型就是成功的：

1. 当前帧 primitive -> cluster 的变化肉眼明显更干净
2. 相邻帧中同一大物体的 track ID 基本稳定
3. 至少能自动挖出若干连续 over-seg 样例
4. strip / video / score matrix 能清楚展示问题
5. 能明确写出下一步该优先改哪一层

---

## 14. 下一步决策规则

### 若 layer-1 明显有效，但 layer-2 经常混淆重复物体
→ 下一步加 second-order relation / graph matching

### 若 layer-1 仍然大量碎片
→ 先改 primitive builder / MP / PT / MT 设计

### 若 layer-1 还不错，但 track birth 太多
→ 优先加强 layer-2 unary feature 与 gating

### 若很多错误都来自 mask 前端
→ 加 oracle mask 模式做对照，再决定是否换 2D front-end

---

## 15. 一句话总结

这版 12h 原型的核心不是追最终精度，而是：

**用最小可运行系统验证 “same-frame 当前观测修复” 是否真的有价值，并通过连续 over-seg 样例可视化，清楚地区分 Layer-1 与 Layer-2 的问题。**
