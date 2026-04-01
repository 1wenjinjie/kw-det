# Stage 0 & Stage 1 详细执行计划

> **项目：** KW-Det — Kinematic Wake-guided Detector  
> **本文档范围：** Stage 0（2026-03-30 ~ 2026-04-12）和 Stage 1（2026-04-13 ~ 2026-04-26）  
> **数据状态（截至 2026-04-01）：**
> - S2SHIPS：16 场景，已完整解压（`dataset_npy/`、`dataset_tif/`、`water_mask/`、`s2ships_labels_mask/`）
> - Finland L2A：18 个产品已全部下载（含所有波段），1 个产品有 1 个 SSL 超时需要补全
> - Finland 标注：5 个 GPKG，8,866 条 boat polygon

---

## Stage 0：环境、数据与协议统一（Mar 30 – Apr 12）

**目标：** 在任何模型代码动工之前，锁定所有不可变协议（split、bands、resample、chip 策略），避免后期因基础设置不同导致实验结果不可对比。

---

### Task 0-A：环境搭建与依赖锁定

**脚本：** `scripts/setup_env.sh`（手动执行，不进入训练流水线）

**步骤：**
1. 确认 CUDA 版本与 PyTorch 版本兼容（RTX 4060 需 CUDA ≥ 11.8）
2. 安装全部依赖并导出精确版本：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics rasterio gdal geopandas pycocotools requests scipy scikit-image
pip freeze > requirements.txt
```

3. 验证核心库可用：

```python
import torch; print(torch.cuda.is_available(), torch.version.cuda)
import rasterio; import ultralytics; import geopandas
```

**Deliverable：** `requirements.txt`（锁定版本）

---

### Task 0-B：数据完整性审计

**脚本：** `scripts/audit_data.py`

#### B1：S2SHIPS 完整性检查

对 16 个 `dataset_npy/*.npy` 逐一执行：

```python
d = np.load(f, allow_pickle=True).item()
assert d['data'].shape == (938, 1783, 12), f"data shape error: {f}"
assert d['label'].shape == (938, 1783, 1), f"label shape error: {f}"
assert d['data'].dtype == np.float64
# 统计每个场景的实例数（连通域）
from scipy.ndimage import label as cc_label
lbl = d['label'][:,:,0]
_, n = cc_label(lbl > 0)
```

预期输出（写入 `data/audit_s2ships.json`）：
```json
{
  "01_mask_rome": {"instances": 67, "labeled_pixels": 4821, "data_nan": 0},
  ...
  "total_instances": 1044
}
```

#### B2：Finland 影像完整性检查

检查 18 个产品目录，对照 `finland_l2a_manifest.json` 验证所有波段文件存在且大小与 HEAD 一致：

```python
for product in manifest:
    for asset in product['assets']:
        path = Path(asset['path'])
        assert path.exists(), f"missing: {path}"
        # 重点核查 B08/B8A/B11（主要波段）
```

**已知问题：** `S2B_35VLG_20220626_0_L2A` 的 coastal band（B01）有一次 SSL 超时，需重新运行：

```bash
python scripts/download_finland_l2a.py \
  --annotations-dir data/finland_2025_annotations \
  --output-dir data/finland_2025_imagery/l2a \
  --workers 1 --timeout 120
```

（脚本本身做了 size-check 跳过已完成文件，只会重试失败的）

**Deliverable：** `data/audit_s2ships.json`、`data/audit_finland.json`

---

### Task 0-C：Finland 目标尺寸可行性评估

**目的：** 确认 Finland 小艇在 10m 分辨率下是否可检测，设定合理的 zero-shot 预期。

**数据来源：** `record_15019034.json` 已记录统计信息：
- 平均面积：5,305 m²，均值直径：92.5 m
- 最小：567.9 m²（约 **2×3 像素**）
- 最大：414,795.7 m²（含 wake 的大船）

**脚本：** `scripts/finland_size_analysis.py`

```python
import geopandas as gpd
import numpy as np

for gpkg in Path("data/finland_2025_annotations").glob("*.gpkg"):
    layers = gpd.list_layers(gpkg)["name"].tolist()
    for layer in layers:
        gdf = gpd.read_file(gpkg, layer=layer)
        areas_m2 = gdf.geometry.area  # CRS 已是 UTM，单位 m²
        side_px = np.sqrt(areas_m2) / 10  # 等效边长（像素）
        print(f"{gpkg.stem}/{layer}: n={len(gdf)}, "
              f"median_side={np.median(side_px):.1f}px, "
              f"<3px: {(side_px < 3).sum()}")
```

**决策规则：**
- 若 `< 3px` 的实例占比 > 30%：zero-shot 评估排除该尺寸段，仅报 `≥ 3px` 子集的指标
- 34VEN + 34VER 是官方测试 tile，需单独统计

**Deliverable：** `data/finland_size_report.md`（含直方图数据，手工写入）

---

### Task 0-D：输入波段与重采样协议锁定

**决策（需在本 task 内确认并写入协议文档）：**

#### 波段选择

| 用途 | 波段 | 原始分辨率 | 训练时分辨率 |
|------|------|-----------|-------------|
| Stream A | B02, B03, B04 | 10 m | 10 m |
| Stream B | B08 | 10 m | 10 m |
| Stream B | B8A, B11 | 20 m | **统一到 10 m** |

#### 重采样策略（推荐：离线预处理，offline）

**采用 offline resample**，原因：
- 训练时无额外开销
- 实验结果完全可复现，不依赖 DataLoader 中的插值实现
- Finland 推理时流程与训练完全对齐

具体方法：双线性插值（`rasterio.Resampling.bilinear`）

```python
# 示例（写入 scripts/resample_bands.py）
import rasterio
from rasterio.enums import Resampling

def upsample_to_10m(src_path, dst_path, target_width, target_height):
    with rasterio.open(src_path) as src:
        data = src.read(
            out_shape=(src.count, target_height, target_width),
            resampling=Resampling.bilinear
        )
        transform = src.transform * src.transform.scale(
            src.width / target_width,
            src.height / target_height
        )
        profile = src.profile.copy()
        profile.update(width=target_width, height=target_height,
                       transform=transform)
        with rasterio.open(dst_path, 'w', **profile) as dst:
            dst.write(data)
```

#### 归一化策略

S2SHIPS 的 `dataset_npy` 值域为 float64，来源是原始 DN 或 reflectance（需实测）。**本项目统一规则：**
- 训练时 per-band 归一化（`(x - mean) / std`），统计量在训练集上计算并固定
- 不使用 Min-Max clip（对 wake 高亮像素不够稳定）

**Deliverable：** `plan/protocol_bands_resample.md`

---

### Task 0-E：S2SHIPS Train/Val 场景级分割

**原则：** 按地理位置做场景级分割，避免同一港口的不同 chip 同时出现在 train 和 val。

**建议分割（固定，不随实验改变）：**

| 集合 | 场景 | 场景数 |
|------|------|--------|
| **Val** | portsmouth, panama, suez6 | 3 |
| **Train** | rome, suez1–5, brest1, toulon, marseille, rotterdam1–3, southampton | 13 |

选择理由：
- `portsmouth`（欧洲英吉利海峡）、`panama`（热带运河）、`suez6`（地中海入口）地理跨度大
- Rotterdam 3 个场景全留 train 以保证训练集船只密度

Finland 完全独立，不参与此分割。

**Deliverable：** `plan/protocol_split.md`（写入 train/val 场景名单，所有后续实验引用此文件）

---

### Task 0-F：Chip 策略与训练量估算

**脚本：** `scripts/estimate_chips.py`（分析用，不做实际裁片）

```python
# S2SHIPS 每个场景 938×1783 像素
# 裁片参数候选：640×640，overlap=0.2（stride=512）

H, W = 938, 1783
chip = 640
stride = 512  # overlap = 128px = 20%

n_row = max(1, (H - chip) // stride + 1)  # = 1
n_col = max(1, (W - chip) // stride + 1)  # = 3

print(f"per scene: {n_row * n_col} chips")  # ~3 chips
print(f"13 train scenes: {13 * n_row * n_col} chips")  # ~39 chips
# 但 ~1044 实例分布不均，rome/rotterdam 密度高
```

**问题：** 39 chips 远远不够（YOLO 建议 ≥ 1500 chips）。

**数据增强策略（针对多光谱，不能套 RGB）：**

| 增强类型 | 操作 | 约束 |
|---------|------|------|
| 几何 | RandomFlip (h/v), RandomRotate90 | 所有波段同步变换 |
| 几何 | RandomCrop（重叠裁片，stride=256） | 保留 ≥ 1 个完整实例 |
| 强度 | per-band 随机 scale [0.8, 1.2] | 各波段独立 |
| 强度 | 随机添加高斯噪声 σ ∈ [0, 0.02] | 模拟海况变化 |
| **禁止** | RGB 色调变换、色彩抖动（HSV/Lab） | 多光谱无意义 |
| **禁止** | Mosaic（不同场景 DN 尺度不对齐） | 需先归一化才能 mosaic |

**调整裁片策略到 stride=256（50% overlap）后重估：**
- per scene：`ceil((938-640)/256+1) × ceil((1783-640)/256+1)` ≈ `2×5 = 10 chips`
- 13 train 场景：≈ 130 chips × 翻转/旋转 ×4 = ~520 有效样本
- 若使用 stride=128，则 ≈ 1040，勉强可用

**建议决策：stride=128，chip=640，并在训练中开启 RandomFlip + RandomRotate90**

**Deliverable：** `plan/protocol_chip.md`（写入 chip 参数与增强策略）

---

### Stage 0 Deliverables 汇总

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 锁定版本的依赖文件 |
| `scripts/audit_data.py` | 数据完整性检查脚本 |
| `scripts/finland_size_analysis.py` | Finland 目标尺寸统计脚本 |
| `scripts/resample_bands.py` | B8A/B11 重采样工具 |
| `scripts/estimate_chips.py` | chip 数量估算脚本 |
| `data/audit_s2ships.json` | S2SHIPS 每场景实例统计 |
| `data/audit_finland.json` | Finland 影像完整性报告 |
| `data/finland_size_report.md` | Finland 目标尺寸可行性分析 |
| `plan/protocol_bands_resample.md` | 波段选择 + 重采样协议（锁定） |
| `plan/protocol_split.md` | train/val 场景级分割方案（锁定） |
| `plan/protocol_chip.md` | chip 尺寸 + 数据增强策略（锁定） |

### Stage 0 Decision Gate

在进入 Stage 1 前，以下三项必须全部确认：
- [ ] 所有 S2SHIPS 场景 npy 格式校验通过，实例总数 ≥ 1000
- [ ] Finland 18 个产品 B08/B8A/B11 文件均完整
- [ ] chip 策略估算的有效训练样本数 ≥ 800（含增强前）

---

---

## Stage 1：S2SHIPS 标签重建与 Wake 伪标签审计（Apr 13 – Apr 26）

**目标：** 从真源数据生成可用于检测训练的统一标签（COCO + OBB），并独立评估 wake 伪标签质量，为后续模型开发提供可信的监督信号。

---

### Task 1-A：S2SHIPS 实例标签重建

**脚本：** `scripts/rebuild_labels_s2ships.py`

#### 输入

- `data/S2SHIPS/dataset_npy/*.npy`（使用 `label` 字段，shape (938,1783,1)，float64）
- `plan/protocol_split.md`（区分 train/val 场景）

#### 处理流程

```
label[:,:,0] > 0
    ↓
scipy.ndimage.label()  →  连通域标记
    ↓
per instance：
    ├─ axis-aligned bbox：(x_min, y_min, w, h)
    ├─ 面积（像素数）
    ├─ 主轴角度 θ：用 cv2.minAreaRect 或 PCA on instance mask coords
    └─ OBB：(cx, cy, w, h, θ)  →  YOLO OBB 格式
```

#### 噪声过滤规则

| 条件 | 处理 |
|------|------|
| 实例面积 < 4 px² | 丢弃（低于 2×2 像素，定位不可信） |
| 实例面积 > 50,000 px² | 标记为 `large_vessel`，单独统计，主任务类别仍为 `ship` |
| 连通域长宽比 > 30:1 | 审查是否为 wake 误入 label，记录 `suspicious=True` |

#### 输出格式

**COCO JSON**（`data/labels/s2ships_coco_train.json` / `s2ships_coco_val.json`）：
```json
{
  "images": [{"id": 1, "file_name": "01_mask_rome_chip_0000.npy", "width": 640, "height": 640}],
  "categories": [{"id": 1, "name": "ship"}],
  "annotations": [{
    "id": 1, "image_id": 1, "category_id": 1,
    "bbox": [x, y, w, h],
    "area": 120,
    "iscrowd": 0
  }]
}
```

**OBB TXT**（Ultralytics OBB 格式，`data/labels/obb/*.txt`）：
```
# 每行：class cx cy w h angle  (归一化到 0-1，angle 单位 rad)
0  0.512  0.489  0.034  0.012  1.047
```

**增强标签 JSON**（`data/labels/s2ships_enhanced.json`）：
```json
{
  "scene": "01_mask_rome",
  "instances": [{
    "id": 1,
    "bbox_xyxy": [x1, y1, x2, y2],
    "obb_cxcywha": [cx, cy, w, h, theta],
    "area_px": 120,
    "suspicious": false,
    "wake_id": null
  }]
}
```

**Deliverable：** `data/labels/s2ships_coco_train.json`、`s2ships_coco_val.json`、`data/labels/obb/`（16 个 scene 的 OBB txt）、`data/labels/s2ships_enhanced.json`

---

### Task 1-B：Wake 伪标签生成

**脚本：** `scripts/generate_wake_pseudolabels.py`

#### 输入

- `data/S2SHIPS/dataset_npy/*.npy`（使用 `data[:,:,7]` → B08，索引从 COCO 元数据或 S2SHIPS 文档确认）
- `data/labels/s2ships_enhanced.json`（Task 1-A 输出，用于 wake-ship 邻近性约束）

> **注意：** S2SHIPS 的 npy `data` 字段中 12 个通道的波段顺序需从数据集文档或 dataset_tif 中的 GeoTIFF 元数据确认，不能假设顺序。确认步骤写入 `scripts/verify_band_order.py`。

#### Pipeline

**Step 1：B08 预处理**
```python
b08 = scene_data[:,:, b08_idx].astype(np.float32)
# 水体掩膜（使用 water_mask）
water = load_water_mask(scene)
b08_water = b08 * water  # 陆地区域置 0
# per-scene 归一化到 [0, 1]
b08_norm = (b08_water - b08_water[water>0].min()) / (b08_water[water>0].ptp() + 1e-6)
```

**Step 2：定向高斯滤波（增强条状结构）**
```python
from skimage.filters import gaussian
# 对 16 个方向（0–168°，步长 12°）分别做 anisotropic Gaussian
# kernel: σ_long=8px, σ_short=1px
# 取 max response + 对应方向
```

**Step 3：Radon 变换**
```python
from skimage.transform import radon
# 在检测到高响应的局部区域（以 ship instance 为中心，crop 128×128）
# angles = np.arange(0, 180, 1)
# sinogram = radon(local_patch, theta=angles)
# 取 sinogram 峰值列 → theta_radon，len_wake（峰值宽度）
```

**Step 4：连通域 + 方向筛选**
```python
# 在定向高斯响应图上做阈值 → 连通域
# 对每个连通域：
#   - 计算主轴方向 θ_wake（via PCA or minAreaRect）
#   - 计算长度 len_wake
#   - 与 Radon 结果比对确认方向一致性（差 < 20°）
```

**Step 5：Wake 置信度评分**
```python
s_wake = (
    0.4 * normalized_response_intensity +
    0.3 * linearity_score +          # 连通域的长宽比归一化
    0.3 * radon_direction_agreement   # 0 or 1（方向一致性）
)
```

**Step 6：过滤规则**
- 保留 `s_wake > 0.7`
- Wake 中心点与最近 ship instance 距离 < `2 × len_ship`
- 长度 `len_wake > 0.5 × len_ship`（太短的不可信）
- 剔除：岸线 buffer 50px 内的线结构、长宽比 < 5:1 的 blob

#### 输出格式

追加到 `data/labels/s2ships_enhanced.json` 的 `wake_id` 字段，并写入：

**`data/labels/wake_pseudolabels.json`**：
```json
{
  "scene": "01_mask_rome",
  "wakes": [{
    "id": "w001",
    "ship_instance_id": 1,
    "theta_wake": 1.047,
    "len_wake": 45.2,
    "s_wake": 0.83,
    "centroid_xy": [312, 489],
    "bbox_xyxy": [290, 460, 355, 520]
  }]
}
```

**Deliverable：** `scripts/generate_wake_pseudolabels.py`、`scripts/verify_band_order.py`、`data/labels/wake_pseudolabels.json`

---

### Task 1-C：Wake 质量审计

**目的：** 独立评估 wake 伪标签精度，决定后续是否可以作为 hard supervision。

#### 审计样本采集

从 `wake_pseudolabels.json` 中按 `s_wake` 分层抽样：

| 置信度段 | 抽样数 |
|---------|--------|
| s_wake ∈ [0.7, 0.8) | 80 |
| s_wake ∈ [0.8, 0.9) | 80 |
| s_wake ∈ [0.9, 1.0] | 80 |

共 **240 个样本**，手工标注为三类：
- `correct`：wake 方向和位置与实际一致
- `ambiguous`：方向有偏差（> 30°）或覆盖不完整
- `wrong`：误检（实际是背景波浪、岸线、反光）

**审计工具：** 将样本渲染为 B08 灰度图 + wake bbox 叠加，保存到 `data/audit/wake_audit_samples/`

**脚本：** `scripts/generate_wake_audit_samples.py`  
**人工标注输出：** `data/audit/wake_audit_labels.csv`（`wake_id, judgment`）

#### 精度统计

```python
df = pd.read_csv("data/audit/wake_audit_labels.csv")
precision_high = (df[df.s_wake >= 0.8].judgment == 'correct').mean()
print(f"High-confidence (s≥0.8) precision: {precision_high:.3f}")
```

#### Decision Gate

| 结果 | 下一步 |
|------|--------|
| 高置信段 precision **≥ 0.80** | Wake 作为 hard supervision，`L_axis` + `L_scale` 全量启用 |
| 高置信段 precision **0.65 – 0.80** | Wake 只作为 attention prior（`M_wake` 可学习，不用作直接角度监督） |
| 高置信段 precision **< 0.65** | Stage 3 中 Stream B 完全依赖 B08/B8A/B11 特征学习，不引入 wake 几何监督 |

**Deliverable：** `data/audit/wake_audit_labels.csv`、`data/audit/wake_quality_report.md`（precision 分布 + 失败模式分析）

---

### Task 1-D：Stage 1 最终标签验证

在进入 Stage 2 前运行 `scripts/validate_labels.py`：

```python
# 验证 COCO 格式合法性
from pycocotools.coco import COCO
coco = COCO("data/labels/s2ships_coco_train.json")
print(f"images: {len(coco.imgs)}, annotations: {len(coco.anns)}")
# 预期：train annotations ≥ 800，val annotations ≥ 150

# 验证 OBB txt 格式
for txt in Path("data/labels/obb").glob("*.txt"):
    for line in txt.read_text().splitlines():
        vals = list(map(float, line.split()))
        assert len(vals) == 6, f"bad OBB line: {line}"
        assert 0 <= vals[1] <= 1 and 0 <= vals[2] <= 1
```

**Deliverable：** `scripts/validate_labels.py`，输出通过即可进入 Stage 2

---

### Stage 1 Deliverables 汇总

| 文件 | 说明 |
|------|------|
| `scripts/verify_band_order.py` | 确认 S2SHIPS npy 波段顺序 |
| `scripts/rebuild_labels_s2ships.py` | mask → COCO + OBB 重建脚本 |
| `scripts/generate_wake_pseudolabels.py` | B08 → wake 伪标签生成脚本 |
| `scripts/generate_wake_audit_samples.py` | 生成人工审计样本图像 |
| `scripts/validate_labels.py` | 标签格式验证脚本 |
| `data/labels/s2ships_coco_train.json` | 训练集 COCO 标签 |
| `data/labels/s2ships_coco_val.json` | 验证集 COCO 标签 |
| `data/labels/obb/` | 16 场景 OBB txt 文件 |
| `data/labels/s2ships_enhanced.json` | 带 wake_id 的增强标签 |
| `data/labels/wake_pseudolabels.json` | 全量 wake 伪标签 |
| `data/audit/wake_audit_labels.csv` | 人工审计结果 |
| `data/audit/wake_quality_report.md` | Wake 质量审计报告 |

### Stage 1 Decision Gate

进入 Stage 2 前必须确认：
- [ ] `s2ships_coco_train.json` 实例数 ≥ 800，val ≥ 150
- [ ] OBB 文件格式验证全部通过
- [ ] `wake_quality_report.md` 已写入 precision 结论，并明确记录对 Stage 3/4 的影响（hard supervision / attention prior / 不用）

---

## 附：脚本目录规划

```
scripts/
├── setup_env.sh                    # Stage 0-A
├── audit_data.py                   # Stage 0-B
├── finland_size_analysis.py        # Stage 0-C
├── resample_bands.py               # Stage 0-D（工具函数）
├── estimate_chips.py               # Stage 0-F
├── verify_band_order.py            # Stage 1-B 前置
├── rebuild_labels_s2ships.py       # Stage 1-A
├── generate_wake_pseudolabels.py   # Stage 1-B
├── generate_wake_audit_samples.py  # Stage 1-C
└── validate_labels.py              # Stage 1-D
```

## 附：协议文档目录规划

```
plan/
├── stage0_stage1_detail.md         # 本文档
├── protocol_bands_resample.md      # Stage 0-D 输出（锁定后不修改）
├── protocol_split.md               # Stage 0-E 输出（锁定后不修改）
└── protocol_chip.md                # Stage 0-F 输出（锁定后不修改）
```
