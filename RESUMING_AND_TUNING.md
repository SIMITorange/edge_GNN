# 检查点恢复与微调指南

## 概述

优化后的配置可能需要更多的调整或监控。本文档说明如何从检查点恢复训练，并如何进行微调。

---

## 1. 从检查点恢复训练

### 1.1 基本恢复脚本

创建 `resume_training.py`：

```python
import argparse
import json
import os
import torch
from config import get_default_configs
from src.data import FourierFeatureMapper, MeshGraphDataset, build_splits, collate_graphs, fit_normalizer
from src.models import build_model
from src.training import CompositeLoss, Trainer
from torch_geometric.loader import DataLoader

def resume_from_checkpoint(checkpoint_path, config_path=None):
    """Resume training from a checkpoint."""
    
    # Load config
    if config_path:
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        # 使用保存的配置（可选）
    else:
        paths, data_cfg, model_cfg, loss_cfg, train_cfg = get_default_configs()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    epoch_from = checkpoint['epoch']
    
    print(f"Resuming from epoch {epoch_from}, checkpoint: {checkpoint_path}")
    
    # 重建模型和其他组件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ... 其他初始化代码（参考 train.py）
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    # 继续训练
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg.epochs,
        checkpoint_path=paths.checkpoint_path,
        save_every=train_cfg.save_every,
        early_stop_patience=train_cfg.early_stop_patience,
        metrics_path=paths.metrics_path,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume training from checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/edge_gnn.ckpt", 
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to saved config snapshot (optional).")
    args = parser.parse_args()
    
    resume_from_checkpoint(args.checkpoint, args.config)
```

### 1.2 恢复训练命令

```bash
# 从最新检查点恢复
python resume_training.py --checkpoint artifacts/edge_gnn.ckpt

# 从特定epoch的检查点恢复
python resume_training.py --checkpoint artifacts/edge_gnn_epoch150.ckpt
```

---

## 2. 动态调整学习率

如果在训练过程中发现学习率不合适，可以：

### 2.1 中止当前训练并调整LR

```python
# 在train.py中临时修改
train_cfg.lr = 1e-4  # 降低学习率
train_cfg.warmup_epochs = 0  # 跳过预热
```

### 2.2 创建LR调整脚本

```python
# adjust_lr.py
import torch
import json

def adjust_checkpoint_lr(checkpoint_path, new_lr):
    """Adjust learning rate in optimizer state."""
    checkpoint = torch.load(checkpoint_path)
    
    # 修改optimizer state中的学习率
    for param_group in checkpoint['optimizer_state']['param_groups']:
        param_group['lr'] = new_lr
    
    # 保存修改后的checkpoint
    out_path = checkpoint_path.replace('.ckpt', '_lr_adjusted.ckpt')
    torch.save(checkpoint, out_path)
    print(f"Adjusted checkpoint saved to {out_path}")

if __name__ == "__main__":
    adjust_checkpoint_lr('artifacts/edge_gnn_epoch150.ckpt', 1e-4)
```

---

## 3. 逐阶段训练策略

如果模型训练不稳定，可以分阶段进行：

### 3.1 第一阶段：快速收敛（无梯度一致性）

```python
# stage1_config.py（复制config.py并修改）
@dataclass
class LossConfig:
    l1_weight: float = 2.0
    relative_l1_weight: float = 0.8
    smoothness_weight: float = 0.08
    gradient_consistency_weight: float = 0.0  # 禁用
    curvature_weight: float = 0.05

@dataclass
class TrainConfig:
    epochs: int = 200  # 只训练200个epoch
    lr: float = 5e-4
```

运行第一阶段：
```bash
python train.py  # 使用原始config，训练200个epoch
```

### 3.2 第二阶段：精细化与物理约束

```python
# stage2_config.py
@dataclass
class LossConfig:
    l1_weight: float = 1.5  # 稍微降低
    relative_l1_weight: float = 0.6
    smoothness_weight: float = 0.10  # 加强平滑
    gradient_consistency_weight: float = 0.25  # 强化物理约束
    curvature_weight: float = 0.08

@dataclass
class TrainConfig:
    epochs: int = 600  # 总共600个epoch（继续400个）
    lr: float = 1e-4  # 降低学习率
    early_stop_patience: int = 50
```

从第一阶段的检查点恢复：
```bash
python resume_training.py --checkpoint artifacts/edge_gnn_epoch200.ckpt
```

---

## 4. 性能诊断与调整

### 4.1 训练曲线分析

使用TensorBoard查看：
```bash
tensorboard --logdir=logs
```

关键指标：

| 指标 | 预期行为 | 问题诊断 |
|------|---------|---------|
| train/loss/total | 持续单调下降 | 如果增加：学习率太高或数据有问题 |
| val/loss/total | 先下后平 | 如果持续下降：欠拟合；快速增加：过拟合严重 |
| train/{target}/l1 | 逐渐接近0 | 如果某个目标下降慢：增加其权重 |
| consistency/grad | 逐渐减小 | 物理约束满足程度 |

### 4.2 根据现象调整

#### 现象1：训练损失不下降
```python
# 可能原因：学习率太小或数据异常
# 解决：
TrainConfig.lr = 1e-3  # 增加学习率
# 或检查数据质量
```

#### 现象2：训练损失波动很大
```python
# 可能原因：学习率太高或梯度爆炸
# 解决：
TrainConfig.lr = 2e-4  # 降低学习率
TrainConfig.grad_clip = 3.0  # 加强梯度裁剪
```

#### 现象3：某个目标拟合特别差
```python
# 可能原因：该目标的数据难度高或权重不够
# 解决方案1：增加该目标的损失权重
# （需要修改losses.py中的权重计算）

# 解决方案2：增加Fourier特征
DataConfig.fourier_features = 32

# 解决方案3：增加隐层维度
ModelConfig.hidden_dim = 512
```

#### 现象4：验证损失快速增加（过拟合）
```python
# 虽然允许过拟合，但可以适度缓解
# 解决方案1：增加dropout
ModelConfig.dropout = 0.15

# 解决方案2：增加平滑项权重
LossConfig.smoothness_weight = 0.15

# 解决方案3：增加weight decay
TrainConfig.weight_decay = 1e-4
```

---

## 5. 超参数扫描策略

如果想系统地优化超参数，可以使用：

### 5.1 简单网格搜索

```python
# grid_search.py
import subprocess
import json

param_grid = {
    'hidden_dim': [256, 512],
    'fourier_features': [16, 32],
    'lr': [3e-4, 5e-4, 1e-3],
    'smoothness_weight': [0.05, 0.10, 0.15],
}

results = []

for hidden_dim in param_grid['hidden_dim']:
    for fourier_features in param_grid['fourier_features']:
        for lr in param_grid['lr']:
            for smoothness_weight in param_grid['smoothness_weight']:
                
                # 修改config
                # ... (使用配置模板)
                
                # 运行训练
                result = subprocess.run(['python', 'train.py'], capture_output=True)
                
                results.append({
                    'hidden_dim': hidden_dim,
                    'fourier_features': fourier_features,
                    'lr': lr,
                    'smoothness_weight': smoothness_weight,
                    'final_loss': extract_loss_from_output(result),
                })

# 保存并分析结果
with open('grid_search_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# 找最优配置
best = min(results, key=lambda x: x['final_loss'])
print(f"Best config: {best}")
```

### 5.2 运行超参数搜索

```bash
python grid_search.py 2>&1 | tee grid_search.log
```

---

## 6. 模型集成策略

如果训练了多个模型，可以组合它们的预测：

```python
# ensemble.py
import torch
import numpy as np

def ensemble_predict(checkpoint_paths, data_loader):
    """Average predictions from multiple models."""
    
    all_preds = []
    
    for ckpt_path in checkpoint_paths:
        model = build_model(...)
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state'])
        
        preds = []
        with torch.no_grad():
            for batch in data_loader:
                pred_dict, _ = model(batch.to(device))
                preds.append(pred_dict)
        
        all_preds.append(preds)
    
    # 对所有预测取平均
    ensemble_preds = {key: [] for key in all_preds[0][0].keys()}
    
    for pred_list in zip(*all_preds):
        for key in ensemble_preds:
            ensemble_preds[key].append(
                torch.stack([p[key] for p in pred_list]).mean(dim=0)
            )
    
    return ensemble_preds
```

---

## 7. 推理优化

### 7.1 模型剪枝（可选）

如果模型太大，可以移除低重要性的参数：

```python
import torch.nn.utils.prune as prune

# 剪枝20%的权重
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.2)
```

### 7.2 量化（可选）

将模型转为半精度（FP16）以加速推理：

```python
model = model.half()  # 转为FP16
```

---

## 8. 保存和分享最终模型

### 8.1 完整模型导出

```python
def save_model_package(model, checkpoint_path, config, output_dir):
    """Save model with config for sharing."""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型
    torch.save(model.state_dict(), f"{output_dir}/model.pt")
    
    # 保存配置
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump({
            'model': vars(config.model),
            'data': vars(config.data),
        }, f, indent=2)
    
    # 保存归一化参数
    shutil.copy(normalization_path, f"{output_dir}/normalization.json")
    
    print(f"Model package saved to {output_dir}")
```

### 8.2 导入和使用

```python
def load_model_package(model_dir):
    """Load model from package."""
    
    # 加载配置
    with open(f"{model_dir}/config.json", "r") as f:
        config = json.load(f)
    
    # 重建模型
    model = build_model(...)
    model.load_state_dict(torch.load(f"{model_dir}/model.pt"))
    
    # 加载归一化器
    with open(f"{model_dir}/normalization.json", "r") as f:
        norm_config = json.load(f)
    normalizer = Normalizer()
    normalizer.load_state_dict(norm_config)
    
    return model, normalizer
```

---

## 检查清单

- [ ] 配置参数已确认
- [ ] 检查点文件存在且未损坏
- [ ] TensorBoard日志可以正常读取
- [ ] 如果调整LR，已保存新的检查点
- [ ] 多阶段训练的配置已准备好
- [ ] 诊断和调整脚本已可用
- [ ] 最终模型已正确保存

---

*更新于 2025-12-01*
