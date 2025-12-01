# 快速参考：核心优化改变

## 🎯 一句话总结
**增加模型容量5倍、弱化正则化、强化损失函数、实现学习率调度** → 充分拟合物理场

---

## 📊 关键参数变化速查表

### 数据配置
```
Fourier特征: 8 → 16 (更精细的空间频率)
Fourier Sigma: 1.0 → 0.5 (更高分辨率)
```

### 模型配置
```
hidden_dim:     128 → 256        (+100%)
num_layers:     6 → 10           (+67%)
heads:          4 → 8            (+100%)
dropout:        0.10 → 0.05      (-50%, 减少正则化)
decoder_hidden: 新增 = 256
```

### 损失函数
```
l1_weight:              1.0 → 2.0   (+100%)
relative_l1_weight:     0.3 → 0.8   (+167%)
smoothness_weight:      0.05 → 0.08 (+60%)
gradient_consistency:   0.1 → 0.15  (+50%)
l2_weight:              1e-5 → 0.0  (完全移除)
curvature_weight:       新增 = 0.05 (Laplacian平滑)
```

### 训练配置
```
epochs:                 200 → 400
early_stop_patience:    30 → 60
lr:                     2e-4 → 5e-4
weight_decay:           1e-5 → 0.0  (完全移除)
grad_clip:              5.0 → 10.0

新增调度:
- use_warmup:          True
- warmup_epochs:       10
- scheduler_type:      'cosine'
- min_lr:              1e-5
```

### 归一化策略
```
vds:                    standard → minmax
ElectrostaticPotential: standard → robust  (对尖峰更鲁棒)
```

---

## 🔧 快速修改定位

| 文件 | 修改内容 | 行数 |
|------|---------|------|
| config.py | DataConfig: fourier参数 | ~50 |
| config.py | ModelConfig: 所有参数 | ~70 |
| config.py | LossConfig: 权重 | ~90 |
| config.py | TrainConfig: 超参 | ~110 |
| train.py | strategy_map 归一化 | ~80 |
| train.py | CompositeLoss初始化 | ~150 |
| train.py | Trainer初始化 | ~165 |
| gnn_model.py | MultiHeadDecoder | 完全重写 |
| gnn_model.py | EdgeGNN类 | 完全重写 |
| losses.py | _laplacian_smoothness | 新增函数 |
| losses.py | CompositeLoss类 | 完全重写 |
| trainer.py | __init__方法 | 大幅扩展 |

---

## 💡 核心优化思想

```
问题: 模型拟合不好物理场
│
├─ 原因分析:
│  ├─ 模型容量不足
│  ├─ 正则化过强
│  ├─ 归一化不当
│  └─ 学习率固定不变
│
└─ 解决方案:
   ├─ 模型: 128→256 hidden, 6→10 layers (5倍参数)
   ├─ 正则: dropout 0.1→0.05, weight_decay 1e-5→0
   ├─ 损失: 主损失权重翻倍, 新增曲率项
   ├─ 归一: robust缩放处理异常, minmax处理相对量
   └─ 优化: 预热+余弦衰减学习率调度
```

---

## 🚀 如何使用优化后的配置

### 开始训练
```bash
python train.py
```

### 监控训练
```bash
tensorboard --logdir=logs
```

### 关键监控指标
1. **train/loss/total** - 应稳定下降
2. **val/loss/total** - 可能轻微上升（过拟合正常）
3. **train/{target}/l1** - 应接近或低于验证集值
4. **consistency/grad** - 指示物理约束满足度

---

## ⚠️ 需要注意的事项

### 显存要求
- 原始模型: ~3-4GB显存
- 优化后模型: **~8-12GB显存** (更深、更宽)
- 如果OOM，降低: hidden_dim (256→192) 或 batch_size

### 训练时间
- 原始: 200 epochs ≈ 2-3小时
- 优化后: 400 epochs ≈ 4-6小时
- 取决于mesh规模和GPU速度

### 数据要求
- 需要足够的训练数据
- 建议 >100 个mesh样本
- 数据噪声会被过拟合放大

---

## 📈 预期训练曲线

```
训练损失 (train/loss/total):
──────────────────────────────
|   \
|    \___
|        \____
|            \_____        ← 最终应接近接近0
0 ─────────────────────────── 200 ─────────────────── 400 epochs

验证损失 (val/loss/total):
──────────────────────────────
|   \
|    \___
|        \____
|            \      ╱╲      ← 可能轻微震荡（过拟合）
|             \    ╱  \
0 ─────────────────────────── 200 ─────────────────── 400 epochs
```

---

## 🔄 如果效果仍不理想的调整步骤

### Step 1: 检查数据
- 验证数据没有异常
- 检查数据标准化是否正确

### Step 2: 增加容量
```python
ModelConfig:
  hidden_dim = 512  (from 256)
  num_layers = 15   (from 10)
```

### Step 3: 调整Fourier
```python
DataConfig:
  fourier_features = 32  (from 16)
  fourier_sigma = 0.3    (from 0.5)
```

### Step 4: 强化特定目标的损失
```python
# 例如，如果ElectricField_x拟合差，
# 在损失函数中给予其更高权重
```

### Step 5: 增加学习时间
```python
TrainConfig:
  epochs = 600  (from 400)
  early_stop_patience = 100
```

---

## ✅ 验证清单

- [ ] 所有Python文件都没有语法错误
- [ ] 新的导入都已添加 (如CosineAnnealingLR等)
- [ ] config.py中所有新参数都已定义
- [ ] train.py中Trainer初始化传入了新的调度参数
- [ ] 损失函数能处理新的curvature_weight参数
- [ ] 第一次运行成功完成(即使只是1个epoch)

---

## 📝 性能基准

假设在相同硬件上运行:

| 配置 | 参数量 | 训练时间 | 预期train loss | 预期val loss |
|------|--------|---------|-----------------|--------------|
| 原始 | 0.5M | 2h | ±0.15 | ±0.18 |
| 优化 | 2.5M | 5h | ±0.05 | ±0.08-0.12 |

*注: 具体数值取决于数据质量和任务难度*

---

## 🎓 为什么这样设计

### 为什么要增加模型容量?
物理场的非线性很强，小模型已饱和，需要更多参数来表示复杂的空间模式。

### 为什么移除L2正则化?
目前的瓶颈是**欠拟合**而非过拟合，L2正则只会让问题更差。

### 为什么用Robust缩放?
物理场经常有异常的尖峰(接面处)，标准缩放会被这些异常值支配，Robust缩放更稳定。

### 为什么增加训练时间?
更大的模型和更复杂的数据需要更多epoch才能收敛。余弦衰减策略会在后期精细调整。

### 为什么使用学习率预热?
避免早期不稳定，让优化器逐步适应，特别是在使用AdamW这样的自适应优化器时。

---

*更详细的说明见 OPTIMIZATION_SUMMARY.md*
