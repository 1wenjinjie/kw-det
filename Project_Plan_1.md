# Project Plan 1 (KW-Det)

## 1. Project Goal

本项目目标不是简单实现一个“双流 YOLO”，而是验证以下核心假设：

Sentinel-2 的多光谱信息，尤其是与水面粗糙度和尾迹相关的波段，可以在薄云遮挡、高海况、极小目标场景下，为船只检测提供比纯 RGB 更稳定的物理先验；这种先验通过 wake-aware fusion + geometry-consistency 能转化成可测的鲁棒性提升。

## 2. Paper Mainline

建议固定论文主线：

1. 光学遥感小目标船舶检测痛点
2. 多光谱尾迹先验
3. 双流融合
4. 几何一致性约束
5. 恶劣海况鲁棒性与跨区域泛化

## 3. Revised Principles

三个核心修改：

1. 先验证伪标签质量，再决定 loss 权重和网络复杂度。
2. 先做强 baseline，再做双流重构，不然增益解释不清。
3. 物理约束改成“轴一致性”而不是“船头方向一致性”，避免 OBB 角度定义带来的错误监督。

方法工作名：

- **KW-Det**
- 全称：**Kinematic Wake-guided Detector for Multispectral Ship Detection in Sentinel-2 Imagery**

---

## 4. Stage 0: Environment, Data, and Protocol Unification

**Time:** 2026-03-30 to 2026-04-12  
**Goal:** 打通环境、统一数据、避免后续实验失真。

### Work Items

- 搭建训练环境：PyTorch + CUDA + Ultralytics + rasterio/gdal + pycocotools
- 打通 Mac -> Windows RTX 3090ti 远程开发与训练链路
- 下载并整理数据：
  - 主训练/验证集：S2-SHIPS
  - 外部泛化集：Finland 2025 coastal small boats dataset
- 明确影像处理协议：
  - 比较 L1C 与 L2A
  - 统一裁片大小、归一化方式、波段命名和缓存格式
- 处理多分辨率问题：
  - B2/B3/B4/B8 为 10 m
  - B8A/B11 为 20 m
  - 必须显式记录重采样策略，禁止隐式混用
- 建立评估协议：
  - 固定 S2-SHIPS 内部训练/验证拆分
  - 芬兰数据集作为严格 zero-shot，不参与调参
  - 额外定义 hard-case subset

### Stage 0 Protocol Locks

- 固定 `S2SHIPS` 场景级 `train/val` 拆分，所有后续实验保持不变：
  - `train`: `brest1`, `marseille`, `portsmouth`, `rotterdam2`, `rotterdam3`, `suez1`, `suez2`, `suez3`, `suez4`, `suez5`, `suez6`, `toulon`
  - `val`: `rome`, `panama`, `rotterdam1`, `southampton`
- 不做实例随机切分，避免同一场景不同 chip 同时出现在 train 和 val 中导致泄露。
- 固定主训练 chip 协议：
  - 主 chip size: `640 x 640`
  - overlap: `160 px`
  - dense sliding stride: `480 px`
  - 为正样本追加 `positive-centered sampling`
  - 每个正样本额外生成 `3` 个抖动 chip，训练时统一 resize 到 `640`
- 以当前固定 split 估算：
  - dense grid 覆盖约 `96` 个 train chips
  - 叠加正样本中心采样后，期望有效训练 chips 约 `2100-2300`
  - 若首次重建标签后有效 chips `< 2000`，则必须提高正样本采样倍率或增强强度
- 固定多光谱重采样策略：
  - 主线使用 `offline resample to 10 m`
  - `B8A/B11` 在预处理阶段采用双线性插值统一到 10 m
  - mask 与标签映射采用 nearest 或 geometry-preserving 方式，不参与双线性插值
  - `online resample` 不作为主线，仅在有余力时做补充对照
- 固定 `Finland` 尺寸可行性审计：
  - 选择 `34VEM / 20220721` 作为 Stage 0 尺寸审计 tile
  - 统计目标在 10 m 栅格上的宽、高和等效像素尺度
  - 固定尺寸分桶：
    - `tiny`: `< 4 px`
    - `small`: `4-8 px`
    - `medium+`: `> 8 px`
  - 若 `tiny` 占比过高，则 Finland zero-shot 需同时报告：
    - 全量结果
    - `detectable subset (>= 4 px)` 结果

### Training Recommendation on RTX 3090 Ti

- 当前硬件按 `RTX 3090 Ti 24GB` 处理，显存不再是主瓶颈，但训练周转时间和 I/O 仍需控制。
- 固定主开发检测器族：
  - 主锚点模型：`YOLO11s-obb`
  - 主线 scale-up 候选：`YOLO11m-obb`
  - 最新模型补充对照：`YOLO26s-obb`
- 训练建议：
  - 单流 `YOLO11s-obb` 基线先用 `imgsz=640`
  - 单流 batch 起始建议 `16-24`
  - 双流模型 batch 起始建议 `8-12`
  - 全程开启 AMP mixed precision
  - 不建议在第一轮直接上 `l/x` 级模型
- 原则：
  - 先用 `YOLO11s-obb` 跑通协议与基线
  - 再用 `YOLO26s-obb` 做最新模型补充对照
  - 若主线稳定，再尝试 `YOLO11m-obb` 冲更高上限

### Deliverables

- 标准化数据清单
- 预处理脚本
- 评估协议文档
- 一批可视化样例（验证配准、重采样、标签质量）
- 固定的场景级 `train/val` split 文档
- 固定的 `chip size + overlap + positive-centered sampling` 协议
- 固定的 `offline resample to 10 m` 预处理说明
- Finland `34VEM / 20220721` 尺寸可行性可视化与尺寸分桶统计

### Decision Gate

- 若 L2A 在海面区域明显不稳定，主实验使用 L1C，L2A 仅作补充对照。
- 若 Finland 中 `tiny (<4 px)` 占比过高，则 zero-shot 结论必须谨慎解释，并把 `detectable subset` 结果作为主文指标之一。

---

## 5. Stage 1: Wake Pseudo-label Generator and Quality Audit

**Time:** 2026-04-13 to 2026-04-26  
**Goal:** 独立建立并评估 wake 伪标签质量，不与模型训练混合。

### Technical Route

- 输入优先 B8（10 m），可对比 B8 + B8A/B11 融合输入
- 在 B8 或融合图上执行：
  - 定向高斯滤波
  - 线结构增强
  - Radon transform
  - 连通域/长度/角度筛选
- 为每个候选尾迹生成：
  - theta_wake
  - len_wake
  - s_wake

### Filtering Rules

- 保留 s_wake > 0.7
- 尾迹与船体必须邻近
- 尾迹长度需在合理范围
- 剔除贴岸线/岛礁边缘/强太阳耀斑区域的孤立直线

### Quality Audit

- 抽查 200-300 个伪标签样本
- 三类标注：correct / ambiguous / wrong
- 统计伪标签 precision（不只做感性观察）

### Deliverables

- 增强标签文件（每条船附带 theta_wake, len_wake, s_wake, wake_present_mask）
- 伪标签质量报告

### Decision Gate

- 若高置信伪标签 precision < 0.8，则 wake 仅作为 attention 引导，不作为监督真值。

---

## 6. Stage 2: Strong Baseline Suite

**Time:** 2026-04-27 to 2026-05-10  
**Goal:** 建立可解释的强基线，回答“增益来自哪里”。

### Required Baselines

1. `Baseline-RGB`
   - 模型固定为：`YOLO11s-obb`
   - 输入：`B2/B3/B4`
2. `Baseline-RGB+B8`
   - 模型固定为：`YOLO11s-obb`
   - 输入：`B2/B3/B4/B8`
   - 首层改为 4 通道输入
3. `Baseline-AllBands-SingleStream`
   - 模型固定为：`YOLO11s-obb`
   - 输入：`B2/B3/B4/B8/B8A/B11`
   - `B8A/B11` 使用 Stage 0 固定的 offline 10 m 重采样版本
4. `Baseline-RGB-OBB`
   - 模型固定为：`YOLO11s-obb`
   - 输入：`B2/B3/B4`
5. `Latest-model Check`
   - 模型固定为：`YOLO26s-obb`
   - 仅在主流程稳定后，复现 `Baseline-RGB` 和最优单流配置做补充对照

### Metrics

- mAP
- Recall
- 小目标召回率
- hard-case recall
- 推理速度
- 显存占用

### Key Questions

- 多光谱是否有显著增益？
- 增益来自波段本身，还是来自双流结构？
- 在相同协议下，`YOLO26s-obb` 是否真的比 `YOLO11s-obb` 带来稳定增益？

说明：若 AllBands-SingleStream 已接近最终方法，论文重点应从“架构创新”转向“物理先验建模”。

### Training Preset

- 单流基线首轮统一使用：
  - `imgsz=640`
  - `batch=16` 作为默认起点
- 若 `YOLO11s-obb` 稳定收敛，再尝试：
  - `batch=24`
  - 或切换到 `YOLO11m-obb`
- 双流架构进入 Stage 3 时：
  - 默认从 `batch=8` 起步
  - 视显存与吞吐再提高到 `10-12`

---

## 7. Stage 3: Two-stream Architecture and Wake-guided Fusion

**Time:** 2026-05-11 to 2026-05-24  
**Goal:** 在基线清楚后引入结构创新。

### Architecture

- **Stream A (RGB):** 输入 B2/B3/B4，使用 `YOLO11s-obb` 级别 backbone 作为主干锚点，建模船体纹理与边缘
- **Stream B (Wake cues):**
  - 优先 B8 主导
  - B8A/B11 可作附加通道（需说明重采样）
  - 使用浅层大感受野模块（空洞卷积或轻量残差块）
  - 不与 RGB 共享最前层权重
  - 深度和宽度控制在 Stream A 的约 `1/2` 级别，优先保证稳定训练与可解释性

### Fusion Strategy

建议从稳定版本起步：

$$
F_{fused} = F_{rgb} \otimes (1 + \alpha \cdot M_{wake})
$$

其中：

- M_wake 优先使用连续注意力图，而非硬二值 mask
- alpha 设为可学习参数，初始化 0.1
- 融合位置优先 neck/FPN 层，避免过早融合

### Must-Prove Points

- Stream B 学到的是水面结构先验，而非重复学习 RGB
- 融合增益主要体现在困难样本，而非普通样本微小提升

### Deliverables

- 双流基线模型
- attention 可视化
- 相对单流 all-band 的增益对比

---

## 8. Stage 4: Geometry Consistency and Joint Training

**Time:** 2026-05-25 to 2026-06-07  
**Goal:** 引入稳定且可解释的几何约束。

### Axis Consistency (Recommended)

原方向一致性：

$$
L_{kinematic} = 1 - \cos(\theta_{ship} - \theta_{wake})
$$

建议改为轴一致性：

$$
L_{axis} = 1 - |\cos(\theta_{ship} - \theta_{wake})|
$$

原因：OBB 主轴无天然船头/船尾定义，直接 cos 会错误惩罚 180° 翻转的正确样本。

### Masked Soft Scale Prior

$$
L_{scale} = 1(s_{wake} > \tau) \cdot \max\left(0, r - \frac{len_{wake}}{len_{ship}}\right)
$$

说明：

- 仅对高置信 wake 样本启用
- r 为经验阈值，不作为死板物理定律
- 目标是抑制明显不合理匹配，而非反演真实航速

### Training Schedule

1. 前半程只训练检测主任务
2. 再解冻全局
3. 最后引入 L_axis + L_scale
4. lambda 从 0.1 warm-up 到 1.0

---

## 9. Stage 5: Robustness, Ablation, and Zero-shot Generalization

**Time:** 2026-06-08 to 2026-06-14  
**Goal:** 完成论文最关键的证据链。

### Hard-case Robustness Test

在 S2-SHIPS 构建困难子集（至少两类）：

- 薄云遮挡
- 高海况白帽/强反光

建议每类 25-50 张，总计 50-100 张。

报告指标：

- Recall
- Miss rate
- 按尺寸分组召回率

### Ablation Study (Minimum)

1. RGB baseline
2. + Stream B
3. + Wake-guided fusion
4. + Geometry losses

可选扩展：

- all-band single-stream
- two-stream without wake pseudo-labels

### Zero-shot on Finland 2025

协议备注需显式写明：

芬兰数据集框可能包含部分 wake，与 S2-SHIPS 标注风格并不完全一致。为避免评估定义差异导致结论失真，建议同时报告：

- 标准 mAP
- Recall
- 中心点命中率或 relaxed IoU 指标
- 按尺寸分桶的结果：
  - `tiny (<4 px)`
  - `small (4-8 px)`
  - `medium+ (>8 px)`
- 若 `tiny` 目标占比较高，则需把 `detectable subset (>=4 px)` 结果单独列出
- 额外保留官方测试区结果：
  - `34VEN`
  - `34VER`

---

## 10. Stage 6: Visualization, Writing, and Draft Convergence

**Time:** 2026-06-15 to 2026-06-21  
**Goal:** 固化故事线，完成初稿，不再大改模型。

### Figures and Tables

- 数据与任务定义图
- 双流结构图
- wake 伪标签生成流程图
- attention/feature map 可视化
- hard-case 检测对比图
- ablation 总表
- zero-shot 结果表

### Writing Structure

1. 问题背景：薄云、高海况、极小目标导致光学遥感船检易失效
2. 观察与动机：多光谱中的 wake/water-roughness 线索可作物理先验
3. 方法：双流结构 + wake-guided fusion + axis-consistency loss
4. 实验：常规精度、鲁棒性、消融、跨区域泛化
5. 讨论：伪标签噪声、分辨率限制、标注协议差异

---

## 11. Final Milestones

1. **2026-04-12**：完成环境、数据协议、场景级 split、chip 协议、重采样策略和 Finland 尺寸审计，稳定复现 baseline 输入
2. **2026-04-26**：完成 wake 伪标签器并获得可接受质量审计结果
3. **2026-05-24**：完成强 baseline 与双流融合主模型
4. **2026-06-14**：完成消融、hard-case、zero-shot 核心实验

---

## 12. Execution Summary (12 Weeks)

- 第 1-2 周：环境搭建、数据整理、场景级 split 固定、chip 协议固定、offline 10 m 重采样策略固定、Finland 尺寸审计、L1C/L2A 对照
- 第 3-4 周：开发 wake 伪标签生成器，完成高置信标签与 200-300 样本质量审计
- 第 5-6 周：建立 `YOLO11s-obb` 的 RGB、RGB+B8、all-band single-stream、RGB-OBB 强基线，并在流程稳定后加入 `YOLO26s-obb` 补充对照
- 第 7-8 周：实现双流结构与 wake-guided fusion，验证相对单流方法增益
- 第 9-10 周：加入 axis-consistency loss 与 masked scale prior，完成联合训练
- 第 11 周：完成 hard-case robustness、ablation、Finland 2025 zero-shot 与尺寸分桶评估
- 第 12 周：完成图表、论文初稿与方法讨论；写作从第 11 周起并行推进
