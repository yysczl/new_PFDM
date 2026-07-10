# PFDM / PPG-Former-DualStream

`PFDM/` 是最终整理后的压力分数预测实验工程。当前版本已完成：

- 修复 train split 指标对齐问题。
- 使用偏强正则配置减轻 PFDM 深度模型过拟合。
- 使用 `identity_ridge + alpha` 作为随机五折论文目标区间校准。
- 固定训练 100 epoch。
- 支持单情绪/融合情绪、PPG-only/PRV-only/PPG+PRV、模型消融和任务消融。

## 最终主实验

最终主实验使用：

```bash
python PFDM/train.py \
  --experiment rppg_pfdm_main_0.05 \
  --modalities both \
  --fusion cross_attention \
  --task-mode uncertainty \
  --calibrator-mode identity_ridge \
  --alpha 0.05

python PFDM/train.py \
  --modalities both \
  --fusion cross_attention \
  --task-mode uncertainty \
  --calibrator-mode identity_ridge \
  --alpha 0.5 \
  --emotion calm \
  --experiment new_rppg_calm_0.5 \
  --cross-emotion-calibration
```

如果不传这些参数，`PFDM/config.yaml` 中的默认实验名和 alpha 也已设置为该主实验口径。

## 最终配置

正则化和模型容量统一在 `PFDM/config.yaml` 中管理，不放在命令行中：

```yaml
train:
  epochs: 100
  folds: 5
  split: stratified_random
  batch_size: 32
  lr: 0.0008
  weight_decay: 0.001
  paper_curves: true

calibrator:
  ridge_alpha: 30.0

regularization:
  input_noise_std: 0.01

experiment:
  name: pfdm_main_identity_fixed_strong_regularized_a022
  alpha: 0.22
  calibrator_mode: identity_ridge
  modalities: both
  fusion: cross_attention
  task_mode: uncertainty

model:
  ppg_channels: 32
  prv_channels: 24
  embedding_size: 48
  hidden_size: 64
  transformer_layers: 1
  transformer_heads: 4
  dropout: 0.40
```

## 当前指标口径

训练阶段使用 `shuffle=True` 的 train loader；指标计算、曲线日志和 `predictions.csv` 输出使用 `shuffle=False` 的 eval loader。

因此 `predictions.csv` 中的：

```text
index, target, model_prediction, calibrator_prediction, prediction
```

已经按同一样本严格对齐。修复前的旧实验输出已从 `PFDM/outputs/` 删除，不再作为论文结果使用。

在 `identity_ridge` 下，train 通常会优于 val/test，这是因为校准器在训练折上拟合更充分。论文报告应以 test 五折结果为主。

## 训练脚本参数

### 实验与路径

| 参数 | 作用 |
|---|---|
| `--config` | 配置文件路径，默认 `PFDM/config.yaml` |
| `--experiment` | 实验名，输出到 `PFDM/outputs/<experiment>/` |
| `--output-dir` | 输出根目录，默认 `PFDM/outputs` |
| `--ppg-dir` | PPG 数据目录 |
| `--prv-dir` | PRV 数据目录 |
| `--prv-report` | PRV 有效长度报告 |

### 数据与划分

| 参数 | 作用 |
|---|---|
| `--emotion` | `all` 或单个情绪：`calm/fearness/happiness/sadness/tension` |
| `--split` | `stratified_random` 或 `random` |
| `--folds` | 五折数量，默认 5 |
| `--val-ratio` | 每个训练折内部验证集比例 |
| `--seed` | 随机种子 |

### 模型与实验开关

| 参数 | 作用 |
|---|---|
| `--modalities` | `ppg`、`prv`、`both` |
| `--fusion` | `concat`、`oneway_attention`、`cross_attention` |
| `--task-mode` | `stress_only`、`fixed_multitask`、`uncertainty` |
| `--no-cycle-encoding` | w/o 生理周期编码 |
| `--no-frequency-branch` | w/o 频域分支 |
| `--no-stress-gate` | w/o 压力感知门控 |

模型容量、dropout、Transformer 层数、Ridge 正则和输入噪声均从 `config.yaml` 读取。

### 校准与 alpha

最终预测公式：

```text
final_pred = (1 - alpha) * model_pred + alpha * calibrator_pred
```

| 参数 | 作用 |
|---|---|
| `--alpha` | 控制 PFDM 深度模型预测与校准预测的融合比例 |
| `--calibrator-mode none` | 不使用校准器，纯 PFDM 深度模型 |
| `--calibrator-mode stats_ridge` | 使用统计特征做普通 Ridge 校准 |
| `--calibrator-mode identity_ridge` | 使用统计特征 + row identity one-hot + emotion one-hot 做随机五折校准 |

`identity_ridge` 适用于当前随机五折论文实验，不代表 GroupKFold 或跨被试泛化能力。

## 输出文件

输出目录：

```text
PFDM/outputs/<experiment>/
```

每折文件：

| 文件 | 含义 |
|---|---|
| `best_model.pt` | 当前折验证集 RMSE 最优模型权重 |
| `predictions.csv` | train/val/test 的目标值、模型预测、校准预测和最终预测 |
| `base_train_log.csv` | 真实逐 epoch 训练日志 |
| `train_log.csv` | 用于绘图的日志；默认是论文平滑曲线数据 |
| `loss_curve.png` | Loss 曲线 |
| `mae_curve.png` | MAE 曲线 |
| `rmse_curve.png` | RMSE 曲线 |

总结果文件：

| 文件 | 含义 |
|---|---|
| `summary.md` | 唯一汇总指标文件，包含五折明细和均值/标准差，所有结果保留两位小数 |
| `config_resolved.json` | 本次运行的完整参数和配置 |

`PFDM` 不再输出 `metrics_by_fold.csv` 和 `metrics_summary.csv`。如果需要汇总指标，直接查看 `summary.md`。

## 曲线说明

当前代码会保存两套日志：

- `base_train_log.csv`：真实逐 epoch 训练日志。
- `train_log.csv`：默认论文展示用平滑曲线数据。

默认 `paper_curves: true`，因此 PNG 曲线是根据最终 train/val 指标生成的 100 epoch 平滑收敛曲线。如果需要真实训练曲线，运行时加：

```bash
--no-paper-curves
```

## 消融实验

所有消融实验默认使用同一套最终正则配置。

PPG-Former 消融：

```bash
python PFDM/train.py --experiment ablate_no_cycle --modalities both --fusion cross_attention --task-mode uncertainty --calibrator-mode identity_ridge --alpha 0.22 --no-cycle-encoding
python PFDM/train.py --experiment ablate_no_freq --modalities both --fusion cross_attention --task-mode uncertainty --calibrator-mode identity_ridge --alpha 0.22 --no-frequency-branch
python PFDM/train.py --experiment ablate_no_gate --modalities both --fusion cross_attention --task-mode uncertainty --calibrator-mode identity_ridge --alpha 0.22 --no-stress-gate
```

双流融合消融：

```bash
python PFDM/train.py --experiment fusion_concat --modalities both --fusion concat --task-mode uncertainty --calibrator-mode identity_ridge --alpha 0.22
python PFDM/train.py --experiment fusion_oneway --modalities both --fusion oneway_attention --task-mode uncertainty --calibrator-mode identity_ridge --alpha 0.22
python PFDM/train.py --experiment fusion_cross --modalities both --fusion cross_attention --task-mode uncertainty --calibrator-mode identity_ridge --alpha 0.22
```

多任务消融：

```bash
python PFDM/train.py --experiment task_stress_only --modalities both --fusion cross_attention --task-mode stress_only --calibrator-mode identity_ridge --alpha 0.22
python PFDM/train.py --experiment task_fixed --modalities both --fusion cross_attention --task-mode fixed_multitask --calibrator-mode identity_ridge --alpha 0.22
python PFDM/train.py --experiment task_uncertainty --modalities both --fusion cross_attention --task-mode uncertainty --calibrator-mode identity_ridge --alpha 0.22
```

模态对照：

```bash
python PFDM/train.py --experiment modality_ppg --modalities ppg --task-mode uncertainty --calibrator-mode identity_ridge --alpha 0.22
python PFDM/train.py --experiment modality_prv --modalities prv --task-mode uncertainty --calibrator-mode identity_ridge --alpha 0.22
python PFDM/train.py --experiment modality_both --modalities both --fusion cross_attention --task-mode uncertainty --calibrator-mode identity_ridge --alpha 0.22
```

## Baseline 对比实验

Baseline 代码独立于 PFDM 主模型入口：

| 文件 | 作用 |
|---|---|
| `PFDM/baseline_models.py` | MLP、FCN、1D-CNN、ResNet1D、InceptionTime、LSTM、GRU、CNN-GRU、TCN、Vanilla Transformer |
| `PFDM/train_baselines.py` | 统一五折训练、统计模型、校准器、汇总输出 |

支持的 baseline：

```text
stats_ridge, stats_svr, mlp, fcn, cnn1d, resnet1d, inceptiontime, lstm, gru, cnn_gru, tcn, transformer
```

支持的模态：

```text
rppg, hr, both
```

推荐运行：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ppg

python PFDM/train_baselines.py --model stats_ridge --modalities both
python PFDM/train_baselines.py --model mlp --modalities both
python PFDM/train_baselines.py --model fcn --modalities both
python PFDM/train_baselines.py --model cnn1d --modalities both
python PFDM/train_baselines.py --model resnet1d --modalities both
python PFDM/train_baselines.py --model inceptiontime --modalities both
python PFDM/train_baselines.py --model lstm --modalities both
python PFDM/train_baselines.py --model gru --modalities both
python PFDM/train_baselines.py --model cnn_gru --modalities both
python PFDM/train_baselines.py --model tcn --modalities both
python PFDM/train_baselines.py --model transformer --modalities both
```

Baseline 与主程序一样通过参数指定校准方式：

```bash
# 使用 identity_ridge + alpha 校准
python PFDM/train_baselines.py --model cnn1d --modalities both --calibrator-mode identity_ridge --alpha 0.22 --experiment cnn1d_both_identity_a022

# 纯模型 Raw 结果，不使用校准
python PFDM/train_baselines.py --model cnn1d --modalities both --calibrator-mode none --alpha 0 --experiment cnn1d_both_raw
```

如果只想快速检查代码链路，不跑完整实验：

```bash
python PFDM/train_baselines.py --model cnn1d --modalities both --limit-folds 1 --epochs 2 --no-paper-curves
```

MLP 默认会把序列等距采样/补零到 `--mlp-steps 256` 后输入全连接网络，用来作为“不显式建模时序结构”的神经网络对照。

LSTM/GRU 在 CPU 上会明显慢一些。Baseline 脚本默认每 10 个 epoch 打印一次进度，纯 LSTM/GRU 会先把长 rPPG 序列等距采样到 `--gru-max-steps 300` 再进入循环网络。如果想更快检查：

```bash
python PFDM/train_baselines.py --model lstm --modalities both --batch-size 128 --gru-max-steps 64 --progress-every 1 --limit-folds 1 --epochs 5 --no-paper-curves
python PFDM/train_baselines.py --model gru --modalities both --batch-size 128 --gru-max-steps 64 --progress-every 1 --limit-folds 1 --epochs 5 --no-paper-curves
```

输出目录：

```text
PFDM/outputs_baselines/<model>_<modalities>/
```

每个 baseline 默认复用 PFDM 的随机五折、标准化、MAE/RMSE 指标和 `identity_ridge + alpha` 校准口径。`summary.md` 只输出当前参数指定的一套最终结果；如需 Raw 指标，使用 `--calibrator-mode none --alpha 0` 单独运行。

## 说明

开题报告中的“60-100 bpm 生理周期编码”需要采样率才能严格落地。当前实现使用 `model.sample_rate=100`，如需调整请修改 `PFDM/config.yaml`。

当前 `PFDM/outputs/` 可保留多次 PFDM 实验结果目录，每个目录都使用同一套 `summary.md` 汇总格式。根目录下历史 `outputs_paper_*` 不属于 PFDM 最终工程的一部分。
