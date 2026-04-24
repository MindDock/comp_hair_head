# CompHairHead

**One-shot Compositional 3D Head Avatars with Deformable Hair**

基于论文 *"One-shot Compositional 3D Head Avatars with Deformable Hair"* (arXiv: 2604.14782) 的开源实现。

## 核心特性

- 🎭 **单张照片重建** — 从一张正面人像生成完整 3D 头部 Avatar
- 💇 **头发/面部解耦** — 独立控制面部表情和头发动态
- 🌊 **物理头发模拟** — Cage-PBD 实现重力驱动的自然头发运动
- ✂️ **发型迁移** — 跨身份的发型交换
- ⚡ **支持 Apple Silicon** — 兼容 M1/M2/M3/M4 (MPS + Metal)

## 技术架构

```
输入图像 → Bald Filter → FaceLift (3DGS) → SAM2 分割 → 非刚性配准
                                                          ↓
                                        G_hair + G_bald + FLAME
                                                          ↓
                                        Cage-PBD 物理模拟 → 实时渲染
```

## 安装

```bash
cd comp_hair_head
pip install -e .

# 或只安装依赖
pip install -r requirements.txt
```

### 外部模型下载

项目需要以下预训练模型（需单独下载）：

| 模型 | 说明 | 下载地址 |
|---|---|---|
| FLAME | 参数化头部模型 | https://flame.is.tue.mpg.de/ |
| SAM2 | 分割模型 | https://github.com/facebookresearch/segment-anything-2 |
| FaceLift | Image→3DGS | https://github.com/FaceLift3D/FaceLift |

将模型放置在 `assets/` 目录下。

## 使用

### 重建

```bash
python scripts/demo.py --image portrait.jpg --output outputs/
```

### 动画驱动

```bash
python scripts/demo.py --mode animate --avatar outputs/avatar.pt
```

### 发型迁移

```bash
python scripts/demo.py --mode transfer --source avatar_a.pt --target avatar_b.pt
```

### Python API

```python
from comp_hair_head.pipeline.reconstruct import ReconstructionPipeline
from comp_hair_head.pipeline.animate import AnimationPipeline

# 重建
pipeline = ReconstructionPipeline()
result = pipeline.reconstruct("portrait.jpg")

# 动画
anim = AnimationPipeline()
anim.setup(result["G_hair_local"], result["G_bald_local"],
           result["flame_model"], result["flame_params"])
frame = anim.animate_frame(expression=expr, pose=pose)
```

## 项目结构

```
comp_hair_head/
├── configs/default.yaml        # 超参数配置
├── comp_hair_head/
│   ├── gaussian/               # 3D Gaussian Splatting
│   │   ├── model.py            # GaussianModel 数据结构
│   │   └── renderer.py         # 可微分渲染器
│   ├── flame/                  # FLAME 参数化模型
│   │   ├── flame_model.py      # FLAME 模型 + LBS
│   │   └── rigging.py          # T_l2g / T_g2l 变换
│   ├── preprocessing/          # 图像预处理
│   │   ├── bald_filter.py      # 去发处理
│   │   └── face_lift.py        # Image→3DGS
│   ├── segmentation/           # 头发分割
│   │   ├── hair_seg.py         # SAM2 分割
│   │   ├── learnable_feat.py   # 可学习特征
│   │   └── boundary_reassign.py
│   ├── registration/           # 配准 & 组装
│   │   └── assembly.py         # 损失函数 & 组件组装
│   ├── dynamics/               # 头发动态 (核心创新)
│   │   ├── cage_builder.py     # Cage 构建
│   │   ├── mvc.py              # Mean Value Coordinates
│   │   ├── pbd_solver.py       # Taichi PBD/XPBD
│   │   ├── collision.py        # Proxy 碰撞约束
│   │   └── hair_deform.py      # 高斯变形传播
│   └── pipeline/               # 端到端管线
│       ├── reconstruct.py      # 重建
│       ├── animate.py          # 动画
│       └── transfer.py         # 发型迁移
├── scripts/demo.py             # CLI 入口
└── tests/                      # 单元测试
```

## 引用

```bibtex
@misc{sun2026oneshotcompositional3dhead,
    title={One-shot Compositional 3D Head Avatars with Deformable Hair},
    author={Yuan Sun and Xuan Wang and WeiLi Zhang and Wenxuan Zhang and Yu Guo and Fei Wang},
    year={2026},
    eprint={2604.14782},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
}
```

## License

MIT
