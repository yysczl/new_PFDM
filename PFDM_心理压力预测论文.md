# 基于 PPG-PRV 双流协同网络的青少年心理压力预测方法

## 摘要

心理压力的客观、连续和无创评估是数字健康与情感计算领域的重要问题。光电容积脉搏波（Photoplethysmography, PPG）具有采集便捷、成本低和适合可穿戴设备部署等优点，但原始 PPG 信号易受个体差异、幅值漂移和运动噪声影响，通用时序模型难以充分利用其周期性和时频特征。为提高基于脉搏信号的压力预测精度，本文提出一种 PPG-PRV 双流协同多任务模型 PFDM。该模型首先通过 PPG-Former 对原始 PPG 波形进行生理周期感知编码、多尺度时频特征提取和自适应分支融合；同时利用轻量级 PRV-Encoder 建模由 PPG 派生的脉率变异性（Pulse Rate Variability, PRV）序列；随后通过双向跨模态注意力实现原始波形形态信息与节律变异信息的深层交互；最后引入多任务学习框架，联合优化压力回归、情绪分类及辅助心理量表预测任务。实验基于包含平静、悲伤、快乐、恐惧和紧张五类情绪条件的青少年脉搏数据展开。结果表明，PFDM 在五折交叉验证测试集上取得 MAE 4.10、RMSE 5.41，优于单独 PPG-Former、PRV-Former 以及简单特征拼接方案。消融实验进一步验证了生理周期编码、频域分支、压力感知门控、双向跨模态注意力和不确定性加权多任务学习的有效性。

**关键词**：光电容积脉搏波；脉率变异性；心理压力预测；跨模态注意力；多任务学习

## 1 引言

心理压力是影响青少年身心健康的重要因素之一。长期处于较高压力水平可能引发睡眠障碍、焦虑、抑郁以及心血管相关风险。传统压力评估主要依赖心理量表或访谈，虽然具有较强解释性，但存在主观性强、滞后性明显和难以连续监测等问题。因此，利用生理信号建立客观、低负担的心理压力评估方法，已成为可穿戴健康监测和情感计算研究的重要方向。

PPG 是通过光学方式测量血液容积变化的生理信号，能够反映心率、血管舒缩和自主神经活动变化。与心电、皮肤电、肌电等信号相比，PPG 采集设备更轻量，更容易集成于手环、腕表和移动健康设备中。已有研究表明，PPG 及其派生的 PRV 特征与压力状态存在关联；压力状态下，交感神经活动增强会影响心跳节律、脉搏波幅度、上升沿形态和频率成分。然而，PPG 信号也具有明显挑战：其波形长、噪声来源复杂、个体差异显著，且压力相关模式往往同时分布在短时形态、长程周期和频域节律中。

现有深度学习方法多采用 CNN、LSTM 或普通 Transformer 对 PPG 序列进行建模。这些方法能够在一定程度上学习时序特征，但通常未显式引入心跳周期等生理先验，也未充分建模原始 PPG 与 PRV 之间的层次化关系。PPG 保留了原始波形细节，PRV 则强调相邻脉搏峰间隔变化，两者来自同一生理过程，却反映不同层次的信息。如何让两类信号在深度特征空间中互补，是提升压力预测性能的关键问题。

针对上述问题，本文提出 PFDM（PPG-Former Dual-Stream Multi-Task）模型。主要贡献如下：

1. 提出面向 PPG 信号的 PPG-Former 编码器，将 60-100 bpm 的心跳周期先验融入位置表示，并结合 Transformer、多尺度卷积和频域摘要提取压力相关特征。
2. 构建 PPG-PRV 双流协同网络，通过双向跨模态注意力实现原始波形与节律变异特征之间的动态交互。
3. 设计多任务学习框架，将压力回归与情绪分类、辅助心理量表预测共同优化，并采用基于同方差不确定性的损失加权策略自动平衡任务贡献。
4. 在五类情绪条件下开展主实验与消融实验，验证模型各组成模块对压力预测性能的贡献。

## 2 相关工作

### 2.1 基于可穿戴生理信号的压力识别

可穿戴压力识别通常基于 ECG、EDA、PPG、皮肤温度、呼吸等信号建立压力状态预测模型。近年来综述研究表明，PPG 与 EDA 是可穿戴压力检测中使用频率较高的生理信号，具有较好的连续监测潜力。Frontiers 2024 年的系统综述进一步指出，压力检测流程通常包含数据采集、预处理、特征计算、模型训练和测试等环节，并且数据采集场景和预处理质量会显著影响模型可靠性。

PPG 在压力识别任务中的优势是非侵入、设备普及和采集成本低；不足是运动伪影、佩戴位置、皮肤状态和环境光都会影响信号质量。传统方法通常从 PPG 或 PRV 中提取心率、峰间期、频域功率和统计特征，再使用 SVM、随机森林等机器学习模型进行分类或回归。此类方法解释性较强，但依赖手工特征，难以端到端捕获复杂的时序模式。

### 2.2 PPG 深度时序建模

随着深度学习的发展，CNN、RNN、LSTM 和 Transformer 被用于生理信号建模。CNN 擅长提取局部波形形态，RNN/LSTM 能够建模时序依赖，但训练效率和长序列表达能力受限。Transformer 通过自注意力机制直接建模序列中任意位置之间的关系，适合捕捉较长范围依赖。近期 TranSenseFusers 将时序卷积与注意力层结合，用于 PPG 及多生理信号压力检测，表明 CNN-Transformer 混合结构适合处理可穿戴 PPG 信号。GPT-PPG 等工作也说明，将 Transformer 类模型迁移到 PPG 信号具有较大潜力。

不过，普通 Transformer 的位置编码来自通用数学形式，未必符合 PPG 的生理周期特征。PPG 波形的核心结构来自心跳周期，压力变化又会影响节律稳定性和波形形态。因此，在模型底层引入心跳周期先验，并同时利用时域、频域和多尺度局部结构，是提升 PPG 压力预测性能的重要方向。

### 2.3 多模态融合与多任务学习

多模态融合能够整合不同来源或不同层次的生理信息。早期融合方法多采用特征拼接或加权平均，但这类静态融合方式难以判断不同模态在具体样本中的有效性。交叉注意力机制允许一个模态以查询形式从另一模态中选择相关信息，适合建模 PPG 与 PRV 这类紧密相关但表示层次不同的信号。

心理压力与情绪状态存在显著关联。多任务学习通过共享底层表示，在相关任务之间迁移知识，有助于缓解小样本和噪声标签带来的过拟合风险。Kendall 等提出的基于同方差不确定性的多任务加权方法，能够让模型自动学习不同任务的相对权重，避免人工设定固定损失比例。本文将这一思想用于压力回归、情绪分类和辅助量表预测的联合优化。

## 3 数据集与预处理

### 3.1 数据来源与标签

实验数据来自青少年 PPG 采集数据。经过质量筛选和完整性检查后，保留 271 个有效样本。每个样本包含 PPG 脉搏波、情绪诱导时间信息和心理量表标签。标签包括压力得分、焦虑得分和抑郁得分，其中压力得分由问卷第 13-19 题计算，焦虑得分由第 20-26 题计算，抑郁得分由第 27-35 题计算。

实验设置包含五类情绪条件：平静（calm）、悲伤（sadness）、快乐（happiness）、恐惧（fearness）和紧张（tension）。根据采集过程中的时间戳，平静片段由第一段稳定时间区间获得；后续较长情绪诱导区间被等分为悲伤、快乐、恐惧和紧张四段。该处理方式使每个有效样本均对应五类情绪条件下的生理片段，为情绪辅助建模和融合情绪训练提供基础。

### 3.2 PPG 与 PRV 构建

原始 PPG 片段长度不完全一致。为满足深度模型输入要求，本文将每个情绪片段重采样为固定长度 3000 点序列。重采样过程中先去除补空位置，再采用插值方法将不同长度波形映射至统一长度，并保留压力、焦虑和抑郁三类标签列。

PRV 序列由 PPG 波形进一步提取。首先对 3000 点 PPG 序列进行带通滤波，然后检测收缩峰位置，计算相邻峰之间的间隔序列。对于异常间隔，使用相邻间隔信息进行修正；随后将峰间期序列归一化并补齐或截断为固定宽度。PPG 与 PRV 在样本、情绪条件和标签上保持一一对应。

除深度序列输入外，模型还使用低维统计特征作为辅助信息，包括 PPG 与 PRV 的均值、标准差、最小值、最大值、分位数和一阶差分波动强度等。这些统计量用于补充深度编码器对全局幅值、分布范围和变化强度的刻画。

## 4 方法

PFDM 的整体结构由 PPG-Former、PRV-Encoder、双流融合模块和多任务预测头组成。设输入的 PPG 序列为 \(X_p \in \mathbb{R}^{T_p \times 1}\)，PRV 序列为 \(X_r \in \mathbb{R}^{T_r \times 1}\)，低维统计特征为 \(s\)。模型输出压力预测值，同时在多任务训练中输出情绪类别和辅助心理量表预测结果。

### 4.1 PPG-Former 编码器

PPG-Former 首先使用两层一维卷积对原始 PPG 序列进行局部特征提取和降采样。卷积核分别覆盖短时脉搏波形局部变化，并将长序列映射为 token 表示：

\[
H_p = \mathrm{ConvStem}(X_p).
\]

为增强模型对心跳周期结构的感知，本文构造生理周期位置编码。设采样率为 \(f_s\)，心跳频率范围对应 60-100 bpm，即 1.0-1.67 Hz。对第 \(t\) 个 token 构造正弦与余弦周期基：

\[
E_t = W_e [\sin(2\pi f_k t/f_s), \cos(2\pi f_k t/f_s)]_{k=1}^{K},
\]

并将其加入 token 表示：

\[
\tilde{H}_p = H_p + E.
\]

随后，Transformer 编码器用于建模长程时序依赖。其输出经过注意力池化得到时域全局特征：

\[
z_t = \mathrm{AttnPool}(\mathrm{Transformer}(\tilde{H}_p)).
\]

考虑到压力相关信息也可能分布在不同局部尺度上，PPG-Former 进一步使用多尺度卷积分支。该分支采用不同大小的一维卷积核提取短、中、长感受野下的波形形态特征，并通过时间平均得到多尺度特征 \(z_m\)。

此外，模型构造频域摘要分支，对 token 序列进行基于余弦基的频率投影，获得反映节律成分强度的频域特征 \(z_f\)。最终，压力感知分支门控根据 \([z_t, z_m, z_f]\) 自适应生成权重：

\[
w = \mathrm{softmax}(W_g [z_t;z_m;z_f]),
\]

并融合各分支：

\[
z_p = \mathrm{LayerNorm}(w_1z_t + w_2z_m + w_3z_f).
\]

当去除频域分支或门控分支时，模型退化为仅基于剩余分支的平均融合，用于消融实验。

### 4.2 PRV-Encoder 编码器

PRV 表示脉搏峰间期变化，是 PPG 的高阶节律特征。由于 PRV 序列长度短、维度低，本文采用轻量级 CNN-Transformer 风格编码器。该编码器由两层一维卷积组成，用于提取局部节律变化：

\[
H_r = \mathrm{Conv}_{PRV}(X_r).
\]

在补齐后的 PRV 序列中，模型使用有效长度掩码避免补齐位置干扰注意力池化。最终通过注意力池化与归一化得到 PRV 全局表示：

\[
z_r = \mathrm{LayerNorm}(\mathrm{AttnPool}(H_r)).
\]

该结构参数量较小，适合建模 PRV 序列的局部变化和全局节律趋势。

### 4.3 双向跨模态注意力融合

PPG 与 PRV 虽来自同一脉搏信号，但信息层次不同。PPG 包含原始波形形态、幅值和局部时频模式；PRV 更直接反映心率节律和自主神经活动变化。为充分利用二者互补性，PFDM 采用双向跨模态注意力。

首先，以 PPG token 为 Query，以 PRV token 为 Key 和 Value，得到由 PRV 节律信息增强的 PPG 表示：

\[
A_{p \leftarrow r} = \mathrm{MHA}(Q=H_p, K=H_r, V=H_r).
\]

其次，以 PRV token 为 Query，以 PPG token 为 Key 和 Value，得到由 PPG 波形形态增强的 PRV 表示：

\[
A_{r \leftarrow p} = \mathrm{MHA}(Q=H_r, K=H_p, V=H_p).
\]

两路注意力输出分别进行时间平均后拼接：

\[
z_{pr} = [\mathrm{Mean}(A_{p \leftarrow r});\mathrm{Mean}(A_{r \leftarrow p})].
\]

相比简单拼接，双向跨模态注意力能够在特征层面实现动态选择，使模型关注与压力变化更相关的跨模态一致信息。

### 4.4 多任务预测头

融合后的双流特征与低维统计特征拼接，输入共享 MLP 表示层：

\[
h = \mathrm{MLP}([z_{pr};s]).
\]

压力预测头输出连续压力得分：

\[
\hat{y}_s = W_s h + b_s.
\]

情绪分类头输出五类情绪概率，辅助预测头输出焦虑和抑郁相关量表得分。压力回归任务采用 Huber 损失，情绪分类任务采用交叉熵损失，辅助回归任务同样采用 Huber 损失。设三类损失分别为 \(L_s\)、\(L_e\) 和 \(L_a\)，多任务总损失定义为：

\[
L = \sum_{i \in \{s,e,a\}} \exp(-\sigma_i) L_i + \sigma_i,
\]

其中 \(\sigma_i\) 为可学习任务噪声参数。该机制能够根据不同任务的学习难度和噪声水平自动调整损失贡献，使辅助任务促进共享特征学习，同时避免某一任务主导训练过程。

## 5 实验与分析

### 5.1 实验设置

实验采用五折交叉验证。每一折中，训练集内部进一步划分验证集，用于选择模型状态。评价指标为平均绝对误差（MAE）和均方根误差（RMSE）。MAE 反映整体平均预测偏差，RMSE 对较大误差更敏感。训练使用 AdamW 优化器，学习率为 0.0008，权重衰减为 0.001，训练轮数为 100，批大小为 32。为提高模型鲁棒性，训练阶段对输入加入小幅随机噪声，并采用梯度裁剪抑制不稳定更新。

### 5.2 基线模型结果

首先使用 CNN+BiLSTM 作为基础对照模型，比较原始 PPG、差分 PPG（dPPG）和 PRV 三类输入在五种情绪及融合情绪条件下的压力预测效果。

**表 1 不同情绪信号及融合情绪的 Baseline-CNN+BiLSTM 实验结果（均值±标准差）**

| Emotion | PPG MAE | PPG RMSE | dPPG MAE | dPPG RMSE | PRV MAE | PRV RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Fear | 5.64 ± 0.37 | 6.57 ± 0.31 | 4.88 ± 0.38 | 5.94 ± 0.43 | 4.59 ± 0.41 | 5.46 ± 0.33 |
| Tension | 5.69 ± 0.35 | 6.61 ± 0.45 | 4.90 ± 0.30 | 5.91 ± 0.31 | 4.66 ± 0.35 | 5.69 ± 0.38 |
| Sadness | 5.75 ± 0.37 | 6.69 ± 0.41 | 4.96 ± 0.29 | 5.98 ± 0.33 | 4.69 ± 0.28 | 5.61 ± 0.36 |
| Happiness | 5.83 ± 0.53 | 6.85 ± 0.55 | 4.98 ± 0.35 | 5.96 ± 0.41 | 4.70 ± 0.38 | 5.63 ± 0.51 |
| Calm | 5.99 ± 0.41 | 7.00 ± 0.46 | 5.01 ± 0.29 | 5.93 ± 0.29 | 4.82 ± 0.22 | 5.78 ± 0.21 |
| Emotional fusion | 5.41 ± 0.24 | 6.53 ± 0.42 | 4.86 ± 0.20 | 5.60 ± 0.20 | 4.56 ± 0.24 | 5.41 ± 0.27 |

可以看到，dPPG 与 PRV 的误差整体低于原始 PPG，说明变化率特征和节律变异特征更容易被简单模型利用。融合五类情绪后，三类输入的误差均有所下降，说明不同情绪条件下存在共享的压力相关生理模式，联合建模有助于扩大样本覆盖并提升共性特征学习能力。

### 5.3 单流模型对比

**表 2 基于 PRV 信号的 PRV-Former 五折实验结果（均值±标准差）**

| Split | MAE | RMSE |
|---|---:|---:|
| Train | 3.90 ± 0.11 | 5.16 ± 0.08 |
| Val | 4.24 ± 0.17 | 5.56 ± 0.22 |
| Test | 4.33 ± 0.17 | 5.61 ± 0.21 |

**表 3 基于 PPG 信号的 PPG-Former 五折实验结果（均值±标准差）**

| Split | MAE | RMSE |
|---|---:|---:|
| Train | 3.63 ± 0.08 | 4.88 ± 0.11 |
| Val | 4.07 ± 0.07 | 5.34 ± 0.05 |
| Test | 4.19 ± 0.18 | 5.45 ± 0.23 |

PRV-Former 在测试集上取得 MAE 4.33、RMSE 5.61，相比基线 PRV 的 MAE 4.56 有一定改善。PPG-Former 在测试集上取得 MAE 4.19、RMSE 5.45，优于 PRV-Former。这说明原始 PPG 波形中仍包含更丰富的压力相关信息，如脉搏上升沿、下降沿、周期稳定性和局部频率变化。通过生理周期编码、多尺度卷积、Transformer 与频域摘要的组合，PPG-Former 能够比简单基线更充分地利用原始波形。

### 5.4 PFDM 主实验

**表 4 基于 PPG 和 PRV 的 PFDM 五折实验结果（均值±标准差）**

| Split | MAE | RMSE |
|---|---:|---:|
| Train | 3.48 ± 0.12 | 4.77 ± 0.14 |
| Val | 4.01 ± 0.13 | 5.38 ± 0.25 |
| Test | 4.10 ± 0.12 | 5.41 ± 0.13 |

PFDM 在测试集上取得 MAE 4.10、RMSE 5.41，优于单独 PPG-Former 和 PRV-Former。该结果说明 PPG 与 PRV 虽具有共同来源，但提供了不同层次的压力相关信息。PPG 分支保留原始波形细节，PRV 分支直接刻画脉搏节律变化，双向跨模态注意力能够让两类特征在深层表示空间中互相补充，从而提高压力预测精度。

### 5.5 PPG-Former 模块消融

**表 5 PPG-Former 模块消融实验结果（均值±标准差）**

| Model | Train MAE | Train RMSE | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|---:|---:|
| w/o 生理周期编码 | 3.75 ± 0.14 | 5.07 ± 0.07 | 4.22 ± 0.19 | 5.54 ± 0.19 | 4.37 ± 0.20 | 5.72 ± 0.23 |
| w/o 频域分支 | 3.85 ± 0.18 | 5.18 ± 0.21 | 4.26 ± 0.04 | 5.62 ± 0.10 | 4.42 ± 0.14 | 5.78 ± 0.24 |
| w/o 压力感知门控 | 3.76 ± 0.13 | 5.09 ± 0.13 | 4.29 ± 0.05 | 5.61 ± 0.12 | 4.38 ± 0.17 | 5.73 ± 0.27 |
| PPG-Former | 3.63 ± 0.08 | 4.88 ± 0.11 | 4.07 ± 0.07 | 5.34 ± 0.05 | 4.19 ± 0.18 | 5.45 ± 0.23 |

完整 PPG-Former 在测试集上的 MAE 为 4.19，优于三个去除模块后的版本。去除频域分支后 MAE 升至 4.42，表明压力信息不仅存在于时域波形形态，也体现在节律与频率成分变化中。去除生理周期编码后 MAE 升至 4.37，说明心跳周期先验有助于模型更快、更稳定地定位 PPG 的周期结构。去除压力感知门控后 MAE 升至 4.38，说明自适应分支融合有助于根据样本特征动态分配不同信息源的重要性。

### 5.6 双流融合消融

**表 6 双流融合消融实验结果（均值±标准差）**

| Model | Train MAE | Train RMSE | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|---:|---:|
| 仅 PPG | 3.63 ± 0.08 | 4.88 ± 0.11 | 4.07 ± 0.07 | 5.34 ± 0.05 | 4.19 ± 0.18 | 5.45 ± 0.23 |
| 仅 PRV | 3.90 ± 0.11 | 5.16 ± 0.08 | 4.24 ± 0.17 | 5.56 ± 0.22 | 4.33 ± 0.17 | 5.61 ± 0.21 |
| 特征拼接 | 3.57 ± 0.12 | 4.83 ± 0.13 | 4.07 ± 0.05 | 5.33 ± 0.11 | 4.16 ± 0.16 | 5.44 ± 0.26 |
| 单向注意力 | 3.87 ± 0.17 | 5.16 ± 0.19 | 4.39 ± 0.13 | 5.74 ± 0.13 | 4.48 ± 0.19 | 5.82 ± 0.19 |
| 双向跨模态注意力 | 3.48 ± 0.12 | 4.77 ± 0.14 | 4.01 ± 0.13 | 5.38 ± 0.25 | 4.10 ± 0.12 | 5.41 ± 0.13 |

双向跨模态注意力取得最优测试 MAE 4.10，优于仅 PPG、仅 PRV 和特征拼接。特征拼接虽然能够同时利用两种输入，但缺少显式的信息筛选和方向性交互，因此提升有限。单向注意力结果较差，说明只允许一种模态从另一种模态获取信息会造成信息流不平衡。双向注意力同时保留 PPG 到 PRV、PRV 到 PPG 的交互过程，更符合原始波形与派生节律特征互补建模的需求。

### 5.7 多任务学习消融

**表 7 多任务不确定性加权消融实验结果（均值±标准差）**

| Model | Train MAE | Train RMSE | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|---:|---:|
| 单任务压力 | 3.51 ± 0.29 | 5.02 ± 0.34 | 4.32 ± 0.27 | 5.96 ± 0.44 | 4.41 ± 0.10 | 6.00 ± 0.16 |
| 多任务固定权重 | 3.47 ± 0.18 | 5.00 ± 0.28 | 4.27 ± 0.19 | 5.93 ± 0.31 | 4.38 ± 0.08 | 5.98 ± 0.14 |
| 不确定性加权 | 3.48 ± 0.12 | 4.77 ± 0.14 | 4.01 ± 0.13 | 5.38 ± 0.25 | 4.10 ± 0.12 | 5.41 ± 0.13 |

不确定性加权多任务模型取得最优测试结果，明显优于单任务压力回归和固定权重多任务模型。单任务模型只关注压力预测，无法利用情绪和辅助心理量表中的关联信息；固定权重多任务虽然引入辅助监督，但不同任务损失尺度和噪声水平不同，固定比例可能导致训练偏向某一任务。自适应任务加权能够在训练过程中动态调整不同目标的影响，使情绪与辅助标签更有效地促进压力表征学习。

## 6 讨论

实验结果表明，PFDM 的性能提升主要来自三方面。首先，PPG-Former 针对 PPG 的生理特性进行结构设计，使模型能够同时关注心跳周期、局部波形形态和频域节律变化。其次，PRV 分支引入了从 PPG 派生出的高阶节律信息，与原始 PPG 波形形成互补。再次，多任务学习利用情绪和心理量表间的关联，为压力回归提供额外监督信号。

从误差指标看，PFDM 的测试 MAE 低于所有单流与静态融合模型，说明跨模态交互确实提升了平均预测精度。RMSE 的改善相对有限，表明少数较大误差样本仍然存在。这可能与个体差异、情绪诱导强度不一致、PPG 信号质量波动以及压力标签本身的主观性有关。后续可考虑引入更严格的跨被试验证、信号质量评估和个体化建模策略，以进一步验证模型在真实应用场景中的泛化能力。

此外，本文模型目前主要使用 PPG 和 PRV 两类生理表示。未来若结合皮肤电、体动、皮肤温度或面部表情等多源信息，可能进一步提高对压力状态的刻画能力。不过，多源传感也会增加采集成本和隐私风险。因此，如何在性能、可穿戴部署复杂度和用户接受度之间取得平衡，仍是后续研究的重要问题。

## 7 结论

本文提出了一种基于 PPG-PRV 双流协同网络的青少年心理压力预测方法 PFDM。该方法面向 PPG 信号的周期性和时频特性，设计了 PPG-Former 编码器；面向 PRV 节律序列，设计了轻量级 PRV-Encoder；通过双向跨模态注意力实现原始波形与派生节律特征的深层交互；并通过多任务学习联合建模压力、情绪和辅助心理量表信息。实验结果表明，PFDM 在五折交叉验证测试集上取得 MAE 4.10、RMSE 5.41，优于单流模型和简单融合策略。消融实验验证了生理周期编码、频域分支、压力感知门控、双向跨模态注意力和不确定性加权多任务学习的有效性。该研究为基于可穿戴 PPG 信号的心理压力评估提供了一种具有应用潜力的深度学习方法。

## 参考文献

[1] Cohen S, Janicki-Deverts D, Miller G E. Psychological stress and disease[J]. JAMA, 2007, 298(14): 1685-1687.

[2] Schneiderman N, Ironson G, Siegel S D. Stress and health: psychological, behavioral, and biological determinants[J]. Annual Review of Clinical Psychology, 2005, 1: 607-628.

[3] Elgendi M. On the analysis of fingertip photoplethysmogram signals[J]. Current Cardiology Reviews, 2012, 8(1): 14-25.

[4] Shaffer F, Ginsberg J P. An overview of heart rate variability metrics and norms[J]. Frontiers in Public Health, 2017, 5: 258.

[5] Schmidt P, Reiss A, Duerichen R, Marberger C, Van Laerhoven K. Introducing WESAD, a multimodal dataset for wearable stress and affect detection[C]//Proceedings of the 20th ACM International Conference on Multimodal Interaction. 2018: 400-408.

[6] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems. 2017.

[7] Kendall A, Gal Y, Cipolla R. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 7482-7491.

[8] Giannakakis G, Grigoriadis D, Giannakaki K, et al. Review on psychological stress detection using biosignals[J]. IEEE Transactions on Affective Computing, 2019, 13(1): 440-460.

[9] Wen Q, Zhou T, Zhang C, et al. Transformers in time series: a survey[J]. arXiv preprint arXiv:2202.07125, 2022.

[10] Taskasaplidis G, Fotiadis D A, Bamidis P D. Review of stress detection methods using wearable sensors[J]. IEEE Access, 2024, 12: 38219-38246.

[11] Bolpagni M, Pardini S, Dianti M, Gabrielli S. Personalized stress detection using biosignals from wearables: a scoping review[J]. Sensors, 2024, 24(10): 3221. https://doi.org/10.3390/s24103221

[12] Pinge A, Gad V, Jaisighani D, Ghosh S, Sen S. Detection and monitoring of stress using wearables: a systematic review[J]. Frontiers in Computer Science, 2024, 6. https://doi.org/10.3389/fcomp.2024.1478851

[13] Kasnesis P, Chatzigeorgiou C, Feidakis M, Gutiérrez A, Patrikakis C Z. TranSenseFusers: a temporal CNN-Transformer neural network family for explainable PPG-based stress detection[J]. Biomedical Signal Processing and Control, 2025, 102: 107248. https://doi.org/10.1016/j.bspc.2024.107248

[14] Chen Z, Ding C, Kataria S, Yan R, Wang M, Lee R, Hu X. GPT-PPG: a GPT-based foundation model for photoplethysmography signals[J]. Physiological Measurement, 2025, 46(5): 055004. https://doi.org/10.1088/1361-6579/add988

[15] Guo R, Yus Kelana B W, Safar A E, et al. Graph attention networks meet transformers: a synergistic approach for multimodal stress recognition from physiological signals[J]. Biomedical Signal Processing and Control, 2026. https://doi.org/10.1016/j.bspc.2025.109206
