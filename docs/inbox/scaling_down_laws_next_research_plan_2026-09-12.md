# Scaling-down laws：下一轮研究执行计划

日期：2026-09-12。依据：Round v3–v8 报告及最新 9 页稿。本文是新的研究计划，尚未执行新 GPU 实验，也未核验 rai 的实时资源与仓库状态。

## 1. 这轮要解决什么

项目定位不变：developing capability-conditioned predictive scaling-down laws，pruning、quantization、distillation 三条平行主臂，预测终点为固定评测分布上的条件 loss。暂不做 MoE、accuracy 网格、新 Qwen 或新增 teacher API。

下一轮的三个交付应当是：

1. **关系的科学价值**：找到能够跨 source 预测响应的可观测输入，或识别蒸馏中能够独立预测训练预算与数据复用效应的关系。复杂形式必须与同输入简单预测器比较。
2. **独立泛化**：在冻结之后测量新模型状态、新 student 配置和新压缩配置。分别报告新配置、尺寸、阶段、家族；不把它们合并成一句“泛化成功”。
3. **决策验证**：在 P/Q/KD 三条路线都有新测量结果的面板上验证选择，特别排除当前 math/code 的选择收益主要来自一个历史 KD 候选的局限。

这轮不是要保证三个复杂公式都胜出，而是提高最有希望的关系的可解释性、可预测性和独立验证强度。

## 2. 开工前：冻结旧证据、预算和测试资源

- 保留全部已有预测与失败记录；旧测试结果可以进入新开发集，但必须标为已揭盲，不能重新叫独立测试。
- 以实际学习参数的身份去重。Pythia-2.8B 的重复 revision 只算一个 source-state；不能仅依赖仓库 revision 名称或某一个文件名。
- 核对实际 GPU 型号、UUID、可用显存和剩余 GPU-h。历史 72 GPU-h 是总额，不自动当成新一轮额度。
- 建议先用不超过约 4 GPU-h 做功能检查、dense 描述符试测及训练吞吐测量；这属于开发，不是确认。
- 剩余预算至少 **30% 预留给独立确认**，最多约 50% 用于新增开发实验，其余用于试测、必要重跑及测量。先压开发规模，不能把确认预算全部花在选模上。
- 对 270M、1B、4B 分别测实际监督 token 吞吐、峰值显存。小模型可以并行；以总吞吐与稳定性决定 lane 数，不以“80GB 能放下”作为并行训练更快的证据。按 UUID 绑定，不干预其他人的任务。
- 先生成开发/确认 manifest。确认模型、配置、池 seed、拟合规则、主要指标和预算必须在新响应揭盲前确定。

## 3. 实验包 A：P/Q 的 source 响应为什么预测不准

### A1. 问题与信息预算

已有 N0、D0、dense loss 的预测器，换 source 后会出现幅度和符号失配。过去的 retention/Wanda 统计已经试过，不能改名重做并作为新发现。

本包检验一个窄假设：**dense 模型输出分布对扰动的敏感性，能否补充平均 dense loss 没有表达的信息？**

明确区分以下输入预算：

| 预算 | 输入 | 新模型需要什么 |
|---|---|---|
| Metadata | N0、已知 D0、方法及配置 | 公开元数据 |
| Dense anchor | 上述输入 + L0,c,j | 一次固定探针 dense 前向 |
| Dense statistics | 上述输入 + 本包两个输出统计量 | 同一轮 dense 前向导出，无反向、无目标压缩结果 |
| K1 | 再加一个预设压缩校准点 | 另计一次压缩和评测成本 |
| Oracle | 目标结果事后拟合 | 仅用于误差诊断 |

正文最终必须说明使用哪档输入。不能把依赖目标压缩结果的系数叫作 basic-input prediction。

### A2. 两个可优先检验的描述符

对每个被评分的 reference token，令 z 为模型最终送入 softmax 的 logits，p=softmax(z)，y 为 reference token。计算：

\[
B_{c,j}=\mathbb E[z_y-\sum_v p_vz_v],\qquad
V_{c,j}=\mathbb E[1-\sum_v p_v^2].
\]

- B 是沿 logit 缩小方向 z→(1−ε)z 时，reference CE 在 ε=0 的一阶变化率；它允许有符号响应。
- V 是 CE 对 logits 的 Hessian 的迹，描述小幅各向同性 logit 噪声下的局部二阶敏感性。
- 使用最终 logits，因此应包含模型既有 softcap 等变换。用稳定的 float32 概率归约，不保存所有 logits。
- 这两个导数关系可直接由 CE 推导；**真实 pruning/quantization 等价于 logit 缩小加各向同性噪声，是待检验假设，不是已有结论。**
- 聚合权重必须与 L 的定义一致。如果主指标是逐样本长度归一化 CE 的均值，B/V 也如此；per-byte 版本按相同逐样本规则重算。

先做一个低自由度诊断候选：

\[
\widehat{\Delta L}_{m,c,j}
=\rho_m(\theta_m)B_{c,j}
+\tfrac12\nu_m(\theta_m)V_{c,j},\quad \nu_m\geq0.
\]

先固定若干配置，检验其系数是否能在 source 之间共享。只有此项有增量，才给 ρ、ν 拟合连续的密度或位宽/粒度响应。逐配置系数模型只验证 source 轴，不能称为压缩强度轴上的 law。

若不能同时跨能力共享，先报告分能力结果；不要为了统一再引入大量交互。

### A3. 开发面板与必要对照

1. 优先对已有、权重身份有效的 Pythia 面板提取 B/V，复用历史 P/Q 响应。首次只选 3 个尺寸 × 2 个训练阶段，随后仅在有增量时扩到其余开发状态。
2. 状态选择依据事先已知的 N、D0、dense loss 覆盖，不按新描述符对目标预测好坏挑选。
3. 第一阶段至少比较：zero；source-free 中位曲线；当前最终预测器；同输入 ridge/低阶响应模型；加入 B/V 的同输入普通回归；上式理论约束候选。
4. 量化加入已有 precision scaling 的 N/D/bit 形式作为适配 baseline。系数仅在本项目开发集重拟，注明原论文量化算法及范围不同。不能把“D0 越大越脆”重新包装成新贡献。
5. 按真实 source-state 留出。对同训练轨迹的多个 checkpoint 单独报告，不把所有配置视为独立 source。
6. 按轻微、非平凡损伤、激进配置分别报告。域划分由开发数据确定；测试发生 cliff 不能剔点。
7. 检查 B/V 与 L0 的共线性及测量稳定性。若新描述符只是 L0 的近重复，不继续增加多项式项来制造增量。

### A4. 量化的一项小型配置控制

只在 A3 有值得继续的预测信号时做：固定 b、g，增加一种明确的量化器变化，例如对称与非对称 grouped RTN，并在开发与测试中记录真实量化步长、零点、裁剪规则及权重舍入误差。

目的是检验名义 bit/group 是否遗漏必要的操作信息。候选相关的权重统计可以由 source 权重计算，但仍须报告计算成本，且不得偷用压缩后的 capability loss。它们属于方法配置的补充输入，不是跨方法通用能力指标。

这只是一个量化器控制；不等价于已经验证 GPTQ/AWQ 等算法迁移。本轮不要同时铺开多个新压缩算法。

### A5. 独立确认与结束条件

- 从 registry 选择至少 4 个未测量压缩结果、权重身份不同的 Pythia source-state，覆盖至少 3 个尺寸；更广的 6-state 面板由预算决定。
- 如历史清单确认从未用过，可保留 SmolLM2-360M 和 SmolLM2-1.7B base 作为新的家族挑战。先只读取允许的 dense 输入，冻结后才测 P/Q 响应。家族迁移、D/N 超出开发范围、tokenizer 改变分别标明。
- 新配置每状态约 3 个 pruning 密度及 4 个量化配置即可；优先能区分候选且损伤非平凡的配置。具体值由开发集确定并冻结，不能按测试结果更换。
- 跨 tokenizer 不直接混合 nats/token 的绝对误差；同时提供 per-byte 对照，原生单位分家族报告。
- 主要指标为 source 等权的 MAE，另报 signed bias、逐 source 误差和预设域内表现。以 source 为 cluster；小 cluster 数时明确区间局限。
- 建议预先定义“有实用增量”为相对最强同信息 baseline 的改善同时超过开发测量噪声和一个事先确定的最小效应；可用约 15% 相对改善作开发推进门槛，但不是投稿标准，也不是看到结果后才设定的阈值。
- 若描述符没有稳定增量，停止该分支。若普通回归打平理论形式，结论是新输入有用，不能声称机制形式被独立确立。若只在已有家族有效，交付范围明确的 law，不抹掉外部失败。

## 4. 实验包 B：蒸馏的 student × 独立数据 × 训练量

### B1. 保持正确的目标和单位

\[
\delta_{c,j}=L_{c,j}(S_{KD})-L_{c,j}(S_0),\qquad
\hat L_{c,j}(S_{KD})=L_{c,j}(S_0)+\hat\delta_{c,j}.
\]

不能回到以大 source 为参照的 log-size 差值，那会重新混入 dense baseline gap。

- DU：池中不同 trace 的监督 completion token 总量。
- T：累计监督 completion token；E=T/DU 是派生的平均复用坐标。
- 另行记录 processed token、optimizer step、学习率轨迹、可训练参数及实际 LoRA 模块。
- 不能把 T、DU、E 当成三个可独立控制的变量；设计的两个独立轴是 DU 与 T。
- teacher、初始 student S0、部署比较的 reference M0 是三个不同角色。Teacher 不必比 student 大；student 若作为压缩候选，则必须满足相对部署 reference 的资源约束。

### B2. 先诊断 4B 的训练协议，避免错把优化差异当尺寸 law

利用 Gemma-3-1B 与 4B 做小 pilot：两个已有开发池，学习率 {5e−5,1e−4,2e−4}，其他 strict LoRA 设置固定；优先复用完全相同协议的数据，最多 12 条短轨迹。

目的不是为每个测试模型调出最优结果，而是判断原先 4B 失败是否主要随优化尺度改变。固定相同 learning rate/rank 并不能自动保证跨尺寸等价训练动态。

- 保持绝对训练时钟，不随池大小或提前保存的 checkpoint 重置 cosine/warmup。
- 记录 dense 前后的同硬件漂移；把训练失败、模式回退、被杀任务视为缺失/失败，不填成有效结果。
- 用开发数据确定一个统一协议，或一个低自由度、可复现且在测试前已冻结的 LR 规则。若需按尺寸调参，公开调参成本，并使所有候选和 baseline 使用同一训练协议。
- 如果 pilot 不能区分协议效应，不声称原因已排除；仍可在固定协议下研究条件关系。

### B3. 真正交叉的开发设计

完整目标矩阵：Gemma-3-270M / 1B / 4B × 三档 DU × 两个独立 pool seed，共 18 条轨迹，每条保存 4 个 T checkpoint。若预算不够，先覆盖三个尺寸与两档 DU 的 12 条轨迹；仅当拟合有实用增量时补第三档。不能先把全预算耗在两个小尺寸上。

- DU 三档建议覆盖约 1:3:9，但由现有 teacher trace 库实际容量确定。
- 同一 pool seed 下采用领域比例固定、长度分层的整条 trace 子集；各 student 使用相同的 trace ID。不要切断推理 trace 来强行精确匹配 token 数。
- DU 在各 student tokenizer 下分别计数，尽可能控制质量和领域构成；报告剩余差异。
- T 的四个水平固定并跨 DU 共享，覆盖首次暴露附近、中等复用和已有证据显示可能产生再损伤的区域。具体预算在 pilot 后、主测试前确定，不根据新测试结果延长。
- 若池尚未被完整访问，保存实际已访问的不同 trace/token 数，不把名义池大小等同于已经提供的训练信息。
- 用 2–4 条角点 train-seed 重复估计优化随机性；pool seed 与 train seed 分开。
- 新组合的 pool 不等于全新 teacher 内容。若底层 traces 已全部进入历史开发，应称“新池组合”，不能称从未见过的训练内容。

### B4. 最多增加两个可解释候选

Baseline 必须包括 zero、分能力常数、T-only、现有单系数 E law、同输入的低阶二维模型。不得只报告相对 zero 的改善。

新增候选应回答“训练获得的变化”和“重复带来的代价”能否分别预测。例如一个待检验的形式是：

\[
\hat\delta_{c,j}
=-a_{c,j}(x_S)\,[1-\exp(-T/\tau)]
+b_{c,j}(x_S)\,[\log(1+T/DU)]^p.
\]

这只是候选：第一项可有符号，第二项是可检验的复用代价假设；δ(0)=0。它允许先改善再恶化，不把所有重复训练强行写成单调有效数据收益。

- 首先固定 p 或只在两个预登记值中选择；不要每能力、每尺寸、每池分别拟合指数。
- source 描述符先分别比较 log NS 与 L0,c,j，一次只允许一个低自由度 source 依赖；先证明需要，再增加第二个输入。
- 先检查 τ、source 斜率、复用项的可识别性。训练区间未出现相应曲率时，报告参数未识别，不外推渐近 floor 或最优预算。
- 与重复数据 scaling 的已有模型比较时，明确迁移自预训练到固定 LoRA trace SFT 的假设；不能把已有重复数据边际递减当作本篇新发现。

### B5. 评价要从“平均拟合误差”升级为预测干预效果

除了 δ 的 MAE，冻结并预测：

1. 固定 DU，加倍 T 会怎样改变每个 c,j 的 loss？
2. 固定 T，扩大 DU 会怎样改变 loss？
3. 若开发数据确实支持非单调，预算最优区间能否在新轨迹上预测？没有识别则不提出此 claim。

先做 leave-one-size-out 与 leave-one-pool-out 开发评价，之后测全新的池组合及未使用的 (DU,T) 配置。4B 已经被反复观察，是新一轮开发尺寸，不能再叫全新 student。

真正的第四尺寸确认（如 Gemma-3-12B）只有在吞吐 profiling 与保留预算允许时做；不能挤掉其他主臂确认。若没有第四尺寸，只声称开发尺寸范围内的新配置预测，尺寸外推仍开放。跨家族小 student 可作为额外挑战，但不能替代同家族尺寸轴的识别。

统计以 pool/完整轨迹为单位；同一轨迹多个 checkpoint 不能独立 bootstrap。报告效应大小、配对 CI、逐尺寸结果和不同预算的 signed bias。不得用六个池的窄 CI 声称模型总体泛化。

### B6. capability 的范围检查与训练共享

- 主指标仍是 loss，不重启 accuracy 网格。
- 新训练输出保存逐样本 sum NLL、token 数、byte 数、sample ID、prompt/reference 版本及分布 j。
- 用同一批新增 checkpoint 同时评测主探针和固定的独立第二分布，新增前向成本即可。对 QA 至少保留 2Wiki 与一个已有外部分布，分别报告。
- 如果两分布的响应不同，可在各自开发 split 内拟合，再预测各自新配置；跨 benchmark 共用系数是更强的单独假设。
- 不要求 QA 必须都改善，不把 2Wiki 改善扩大为一般 QA 能力提升，也不因外部响应不同就判主探针无效。

## 5. 实验包 C：真正独立的三路线选择

本包在 A/B 的预测器选择完成后再运行。先锁定候选规则，再测新响应；它的目的不是把已有 map 画得更好看。

### C1. 小而完整的候选面板

目标为至少三个部署 reference，覆盖两个家族；每个 reference 都有 P、Q、KD 以及 dense-small-student 对照。

可用构成（以 registry 未使用清单和预算为准）：

- 一个新的 Pythia reference 阶段，对应同阶段较小 student；
- Gemma-3-4B reference，对应 1B 或 270M student，新池与新预算训练；
- 若 SmolLM2 确认为新家族，可用 1.7B reference 与 360M student。

已有 reference checkpoint 上的未测配置，叫新配置；从未测过的 reference 才叫新 source。不要混称。

每个案例约 3 个 pruning 配置、4 个 quantization 配置、2 个 KD 预算。KD 至少两个新 pool seed，必须是本轮冻结之后新训练的 adapter。可复用 A/B 的独立确认候选，但只有在选择预测也于其测量前冻结时，才算前瞻选择测试。

### C2. 强对照与公平性

对每个预算使用完全相同的候选配置和成本定义，比较：

- 新 law 选择器；当前锁定规则；quant-only；prune-only；distill-only；最便宜可行；未蒸馏的小 dense student；实测 oracle。
- 不可行就是不可行，不回退到超预算 reference。
- 报各方法覆盖率，以及共同可行格上的公平比较。
- 尽量保持原来的名义矩阵存储指标，以隔离预测改进；mask、scale、zero-point 等被省略的开销明确列出。不因此声称实际延迟最优。
- student/reference 不同 tokenizer 时，先统一评测单位，否则不能直接按绝对 nats/token 排序。
- 单能力主目标与多能力 max ΔL 的定义、ties、候选集规则在测前冻结。

### C3. 主要报告

1. 每个 reference 的 regret 和相对 quant-only 的配对差，再给 source 等权汇总。
2. 去掉任意一个 reference 的敏感性；去掉任意一个 KD 候选的敏感性。不能让一个旧 student 决定全部结论。
3. 是否选对“保留原始小模型”还是“训练后小模型”，说明 KD 本身是否提供实用收益。
4. 精确配置与方法命中分开；候选集覆盖率同时给集合大小。宽集合包含 oracle 不是准确选中 oracle。
5. budget cells 不是独立样本。小 reference 数报告逐案例结果，不靠数百格制造过窄 CI。

若新法与 quant-only 持平，应明确其适用性；不能为了制造方法多样性而事后挑预算、能力权重或删量化配置。若两种非量化路线在冻结的案例中确实有优势，并且选择器提前识别，才是更强的决策贡献。

## 6. 继续、停止与论文怎么变化

| 观察结果 | 下一步 | 可增加的论文结论 |
|---|---|---|
| B/V 等少量 dense 统计跨 source 有稳定增量 | 做新状态/家族确认，拟合连续强度响应 | 哪些压缩前信息支持能力响应预测 |
| 新输入有效，但普通回归与理论式打平 | 选更简单者，保留理论式为解释候选 | 输入有效；不确认特定机制形式 |
| P/Q 仍只能靠中位曲线预测 | 停止堆 descriptor/多项式，保留该适用域 | 当前信息预算下的可预测边界 |
| KD 在多个尺寸上预测 DU/T 的干预效果 | 冻结后验证新池/第四尺寸或家族 | 条件明确的 student–data–training response law |
| KD 只有单一尺寸或分布成立 | 写清范围，仍做决策检验 | 局部可复用关系，不称联合普适律 |
| 新三路线选择收益跨多个 reference 复现 | 将其作为正文主结果 | 预测关系有可独立复现的决策价值 |
| 决策仍主要来自一个候选 | 明确个案范围，不追加阈值搜寻 | 实用案例，不能宣称广泛最优 |

所有结果都可进入 investigation，但只有跨 source 或跨配置的独立、实用增量，才能升级“predictive law”的主 claim。新增实验不保证接收率提高；目标是增加审稿人能够核验的新知识。

## 7. 执行顺序与汇报格式

先做资源/协议审计；A 的 dense 描述符提取与 B 的短 LR pilot 可并行；然后根据开发信号确定主矩阵，最后运行保留的独立确认和 C。不要先跑完所有 test 再决定 headline。

每个阶段结束主动更新 RESULTS_LEDGER、CLAIM_EVIDENCE_MATRIX、EXPERIMENT_MANIFEST 和本轮报告。长任务每 2 小时报告一次实际状态；阻塞、失败、协议偏离及预算异常立即报告。交互式分析期间给简短进度，避免仅在一天结束才汇总。

每次汇报固定包含：

1. 本批研究问题与此前未排除的解释。
2. 实际 source/权重身份、模型阶段、池/训练 seed、配置、计量单位。
3. 冻结时间、commit、允许输入及校准成本。
4. 已完成/失败/待运行数量，GPU 型号、耗时、预算余额。
5. 候选与最强同信息 baseline 的 MAE、配对差、区间、逐 source/轨迹结果。
6. 原假设获得支持、未获支持或无法识别；解释范围和反例。
7. 下一步继续/停止理由，更新了哪些正文、表图和 ledger 条目。
8. JSON 路径、生成脚本、复现命令；不手抄或覆盖旧冻结预测。

## 8. 相关工作：此次需要真正超过什么

- [Scaling Laws for Precision](https://arxiv.org/html/2411.04330v2)：已有 N、训练数据与量化精度的联合关系，并报告训练量增加后的 PTQ 敏感性。我们需要检验能力/分布条件化及新 source 预测的额外价值，而不是仅复现训练量效应。
- [Distillation Scaling Laws](https://proceedings.mlr.press/v267/busbridge25a.html)：已有 student size、训练数据和 teacher performance 的联合建模。其 logit distillation 设置与本项目 trace LoRA SFT 不同；相关结构可作有范围说明的候选，不直接照搬系数。
- [Scaling Data-Constrained Language Models](https://arxiv.org/html/2305.16264v5)：重复数据与有效数据量已有 scaling-law 研究。新增贡献应是能力条件下的实际响应、尺寸/优化协议条件，以及独立预测的预算决策。
- [P² Law](https://aclanthology.org/2025.acl-long.283/)：讨论剪枝后的 post-training scaling。与本轮无恢复 pruning 不能直接当作完全同设置的 baseline；要在 related work 清楚区分。
- 新家族候选的官方信息：[SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)、[SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B)。使用 base 版本，核验实际参数和训练 token 元数据；它们与 Pythia 训练范围不同，不把跨范围挑战称作单纯插值。
