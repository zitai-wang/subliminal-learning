# Subliminal Learning for VLM Reasoning

研究能否通过 Teacher 生成的纯数字序列，将目标推理行为传递给视觉语言模型 Student，并改善其在 MathVista 上的表现。

当前实验使用 `Qwen2.5-VL-7B-Instruct`。Teacher 由固定系统提示构建；Student 从相同原始基座开始，仅用数字请求与数字回答进行 LoRA 训练。MathVista 只用于评测。

## 实验流程

完整流程对应 [step.md](step.md) 中的 7 个步骤。

| 步骤 | 操作 | 主要产物 |
|---|---|---|
| 1 | 评测原始模型的直接回答和 CoT 表现 | `baseline_direct`、`baseline_cot` |
| 2 | 添加目标系统提示，评测 Teacher 的 CoT 表现 | `teacher_cot` |
| 3 | 随机生成数字前缀，让目标 Teacher 续写约 30,000 条 | `data/teacher_raw.jsonl` |
| 4 | 过滤数字回答，随机选择 10,000 条 | `data/train.jsonl`、`data/summary.json` |
| 5 | 从原始模型开始训练 Student，训练 6 轮 | `checkpoints/epoch_06/` |
| 6 | 用与原始模型相同的设置评测 Student | `student_direct`、`student_cot` |
| 7 | 校验完整性、统一计分并比较各组结果 | `results.md`、`results.json` |

Teacher 是否建立了准确率优势，以及这些优势能否通过数字数据传递，均由实验检验。当前是单 seed、单目标 Teacher 的流程验证实验；完整设置见 [ROUTE1_RUN.md](ROUTE1_RUN.md)。

## 数据与训练设置

数字请求中的开头序列由程序随机生成，续写部分必须由目标 Teacher 生成。例如：

```text
User: Continue this sequence: 182, 818, 725.
      Return only up to 10 additional integers between 0 and 999,
      separated by commas. Do not explain your answer.
Assistant: 629, 937, 483, 762, 519, 674, 838, 291
```

过滤要求回答正常结束，只含 1–10 个 0–999 的整数，使用一致的数字分隔符；允许括号及结尾句号。先采样 30,000 条，通过过滤的数据不足 10,000 条时继续采样，最多 60,000 条。最终使用 seed 42 随机选择 10,000 条，记录数据 SHA256。

Student 训练数据只保留 `id`、`user`、`assistant`。目标 Teacher 的系统提示不进入 Student 数据；Student 使用普通助手系统提示。训练不读取 MathVista，不包含图像、目标任务题目、答案或 CoT。

| 参数 | 当前设置 |
|---|---|
| 基座 | Qwen2.5-VL-7B-Instruct，BF16 |
| 可训练模块 | 语言 decoder 的 `q_proj`、`k_proj`、`v_proj`、`o_proj` LoRA |
| 视觉模块及其余基座参数 | 冻结 |
| LoRA | rank 16，alpha 32，dropout 0.05 |
| 优化器与学习率 | AdamW，`1e-4`，weight decay 0 |
| 学习率调度 | 3% warmup，然后线性衰减 |
| 微批与梯度累积 | 4 × 16，有效 batch size 64 |
| 最大序列长度 | 256；超长样本报错，不静默截断 |
| 训练轮数 | 6，保存第 1、3、6 轮 |
| 主结果 | 提前固定 epoch 6，不根据 MathVista 分数选检查点 |
| 损失 | 只监督数字回答与 EOS；按累积组的有效回答 token 归一化 |
| 稳定性与显存 | 梯度裁剪 1.0、梯度检查点、FlashAttention 2 |
| 恢复 | 每 50 次优化器更新及每轮结束保存训练状态与随机数状态 |

正常情况下每轮约 157 次优化器更新，6 轮共约 942 次。OOM 恢复时如需改变微批大小，应相应调整梯度累积，保持有效 batch size，并保存实际运行设置。

## 评测设置

使用完整 MathVista `testmini`，共 1,000 题。图像直接从 parquet 读取，`min_pixels=3,136`、`max_pixels=1,003,520`。

| 条件 | 模型 | 系统提示 | 输出要求 | 最大新 token |
|---|---|---|---|---:|
| `baseline_direct` | 原始模型 | 普通助手提示 | 只输出最终答案 | 512 |
| `baseline_cot` | 原始模型 | 普通助手提示 | 分步推理及最终答案 | 2048 |
| `teacher_cot` | 原始模型 | 目标推理行为提示 | 分步推理及最终答案 | 2048 |
| `student_direct` | 原始模型 + epoch 6 LoRA | 普通助手提示 | 只输出最终答案 | 512 |
| `student_cot` | 原始模型 + epoch 6 LoRA | 普通助手提示 | 分步推理及最终答案 | 2048 |

各组使用 greedy 解码；当前评测 batch size 为 4，采用左侧填充。所有逐题原始回答均保存到 JSONL。

用户在查看初轮评测及截断诊断后，将 direct/CoT 上限从 128/512 改为 512/2048。新预算使用独立运行目录重跑全部评测，旧结果保留；数字数据与六轮训练设置不变。这是同一评测集上的后续实验，报告保留协议修订记录。

计分使用本地确定性答案提取与归一化，不调用外部裁判。最终汇总会统一重新计分、检查 1,000 个题目 ID 是否一致，并保留计分修正记录。复杂 LaTeX 或不遵循输出格式的回答仍可能无法正确提取，因此报告应注明本地计分口径。

主要比较 `student_direct − baseline_direct` 和 `student_cot − baseline_cot`；Teacher 的提示效果通过 `teacher_cot − baseline_cot` 检查。报告提供准确率变化、改对/改错题目数，以及配对问题 bootstrap 区间。这些区间不表示训练 seed 之间的波动；当前没有中性 Teacher 数字训练对照，分数变化不能单独证明因果迁移。

## 环境与数据准备

需要支持 BF16 和 FlashAttention 2 的 CUDA 环境。当前执行环境记录如下：

| 依赖 | 版本 |
|---|---|
| Python | 3.10.18 |
| PyTorch | 2.9.1 |
| Transformers | 4.57.3 |
| PEFT | 0.14.0 |
| FlashAttention | 2.8.3 |
| PyArrow | 21.0.0 |
| NumPy | 2.2.6 |
| PyYAML / Pillow / Levenshtein | 6.0.2 / 10.4.0 / 0.27.3 |

`requirements.txt` 和 `environment.yml` 保留了既有实验的依赖记录；运行当前路线一时，以这里记录的环境及各次运行的 `provenance/environment.json` 为参考。

准备本地模型权重和 MathVista `testmini` parquet，并修改 [src/config/subliminal_route1.yaml](src/config/subliminal_route1.yaml) 中的两个路径：

```yaml
model_path: /path/to/Qwen2.5-VL-7B-Instruct
mathvista_parquet: /path/to/mathvista/testmini.parquet
```

模型加载使用 `local_files_only=True`；运行脚本也开启了 Hugging Face 离线模式。模型和数据需要提前准备。

当前环境的 AutoAWQ 与 Transformers 不兼容。BF16 路线在 LoRA 创建和加载时跳过 PEFT 对 AutoAWQ 的无用探测，继续使用标准稠密 Linear LoRA；已用小模型验证训练、保存及重载一致性。

## 运行完整流程

从项目根目录执行。`RUN_DIR` 应为本次实验独立的输出目录：

```bash
SUBLEARN_PYTHON=python SUBLEARN_GPU=2 \
  bash scripts/run_subliminal_route1_cuda2.sh "$PWD/runs/route1_example"
```

可以通过环境变量选择 Python、物理 GPU 编号及评测 batch size；默认 Python 为本机 `licv` 环境，默认 GPU 为 2，默认评测 batch size 为 4。启动前默认每 15 秒检查指定 GPU，确认没有计算进程后才加载模型。`SUBLEARN_GPU_POLL_SECONDS=2` 可将检查间隔改为 2 秒。等待状态保存在 `status.json`。

```bash
export SUBLEARN_PYTHON=/path/to/python
export SUBLEARN_GPU=2
export SUBLEARN_EVAL_BATCH_SIZE=4
bash scripts/run_subliminal_route1_cuda2.sh "$PWD/runs/route1_example"
```

长时间运行可放在 tmux 中：

```bash
tmux new -s sublearn
# 在 tmux 中执行上面的完整流程命令；Ctrl+b，再按 d 可退出终端连接。
```

也可以分别执行各阶段：

```bash
export CUDA_VISIBLE_DEVICES=2
RUN_DIR=runs/route1_example
CONFIG=src/config/subliminal_route1.yaml
python src/subliminal_route1.py evaluate --config "$CONFIG" --run-dir "$RUN_DIR" --conditions baseline_direct baseline_cot
python src/subliminal_route1.py evaluate --config "$CONFIG" --run-dir "$RUN_DIR" --conditions teacher_cot
python src/subliminal_route1.py generate --config "$CONFIG" --run-dir "$RUN_DIR"
python src/subliminal_route1.py train --config "$CONFIG" --run-dir "$RUN_DIR"
python src/subliminal_route1.py evaluate --config "$CONFIG" --run-dir "$RUN_DIR" --conditions student_direct student_cot
python scripts/summarize_subliminal_route1.py --run-dir "$RUN_DIR"
```

同一运行目录会保存有效配置，并检查后续命令的配置是否一致。生成可接续已有原始记录，训练可恢复保存的训练状态，评测会跳过已有题目。

已有完整数字训练数据时，可在另一张空闲卡并行训练，主流水线仍执行评测：

```bash
SUBLEARN_GPU=1 bash scripts/run_subliminal_training_worker.sh "$PWD/runs/route1_example"
```

训练进程每 2 秒检查指定 GPU，空闲后开始。文件锁保证同一运行只训练一次；主流水线到达训练步骤时等待已有训练完成，再使用 epoch 6 评测。独立训练进度在 `training_status.json`，等待状态在 `gpu1_training_queue.json`。

无人值守运行时，可以在另一 tmux 会话中启动监控：

```bash
python scripts/watch_subliminal_route1.py --run-dir "$PWD/runs/route1_example"
```

监控每 30 秒记录状态。遇到已知 CUDA OOM 时，它检查空闲 GPU，按条件使用空闲双卡分配模型，或减小微批并保持有效 batch size 64，然后恢复流程；最多自动恢复 4 次。其他错误写入待排查状态。运行与恢复设置保存在 `supervisor_status.json` 和 `provenance/oom_recovery_events.jsonl`。

## 查看进度和结果

```bash
RUN_DIR=runs/route1_example
cat "$RUN_DIR/status.json"
tail -f "$RUN_DIR/logs/step1_baseline.log"
# 训练开始后：
tail -f "$RUN_DIR/logs/step5_train.log"
```

输出目录结构：

```text
RUN_DIR/
├── config.json                     # 有效配置
├── status.json                     # 当前阶段、计数、准确率或训练 loss
├── pipeline_exit.json              # 全流程退出码
├── data/
│   ├── teacher_raw.jsonl           # Teacher 原始输出及过滤标记
│   ├── train.jsonl                 # 10,000 条 Student 训练数据
│   └── summary.json               # 过滤统计与数据 SHA256
├── checkpoints/
│   ├── epoch_01/
│   ├── epoch_03/
│   ├── epoch_06/
│   └── trainer_latest.pt          # 优化器、调度器及随机数等恢复状态
├── evaluation/
│   ├── baseline_direct.jsonl
│   ├── baseline_cot.jsonl
│   ├── teacher_cot.jsonl
│   ├── student_direct.jsonl
│   ├── student_cot.jsonl
│   └── *_summary.json
├── logs/                           # 按 step 命名的阶段日志
├── provenance/                     # 代码、环境与执行记录
├── training_preflight.json         # 可训练参数与数据检查
├── train_metrics.jsonl             # loss、梯度范数、学习率、显存等
├── results.md                      # 七步完成后的可读结果
└── results.json                    # 完整指标与配对比较
```

本机当前运行位置可查看 `runs/latest_route1.json`。运行数据、日志、检查点、权重、缓存与本地凭据由 `.gitignore` 排除，不随代码上传。

## MathVision 跨数据集评测

完成 MathVista 主实验后，可以复用完全相同的数字训练数据和 epoch 6 Student，在 MathVision `testmini` split（304 题）上评测五个条件。这样只改变评测集，不重新训练 Student，能够检查阴性结果是否依赖 MathVista。

```bash
SUBLEARN_GPU=1 bash scripts/run_subliminal_mathvision.sh \
  "$PWD/runs/mathvision_example"
```

MathVision 配置位于 `src/config/subliminal_mathvision.yaml`。评测沿用 direct 512 / CoT 2048、greedy decoding 和 batch size 4；选择题使用确定性的选项映射，开放题使用 boxed 答案提取及符号等价判断，不调用远程裁判。结果保存在新运行目录的 `results.md`、`results.json` 和 `evaluation/*.jsonl`。

## 多基准交叉数据集复核

还可以在 ScienceQA、MMMU、ChartQA、MMBench、MathVerse 和 MMStar 上复用同一份数字训练数据与固定 epoch 6 Student。各基准使用全部有答案的评测 split：ScienceQA test 4,241 题、MMMU validation 900 题、ChartQA test 2,500 题、MMBench dev 4,377 题、MathVerse testmini 3,940 题、MMStar val 1,500 题。所有条件仍使用 greedy decoding，direct 上限 512 tokens，CoT 上限 2048 tokens。

GPU 队列会在指定显卡连续两次被检测为空闲后领取下一个数据集；单次 OOM 或运行失败会以 batch size 2 自动续跑一次，已有 JSONL 行会被保留并跳过：

```bash
python -u scripts/run_subliminal_benchmark_queue.py \
  --run-root "$PWD/runs/cross_dataset_suite" \
  --gpus 0,1,2,3
```

每个子目录都会生成五种条件的逐题输出、日志、`results.md`、`results.json` 和配对 bootstrap 区间。配置位于 `src/config/subliminal_{scienceqa,mmmu,chartqa,mmbench,mathverse,mmstar}.yaml`，统一执行器与汇总器分别是 `src/subliminal_mcq.py` 和 `scripts/summarize_subliminal_mcq.py`。

## 代码组织

- [src/subliminal_route1.py](src/subliminal_route1.py)：当前路线一的生成、训练和评测入口。
- [src/config/subliminal_route1.yaml](src/config/subliminal_route1.yaml)：提示与实验参数。
- [scripts/run_subliminal_route1_cuda2.sh](scripts/run_subliminal_route1_cuda2.sh)：七步流水线。
- [scripts/summarize_subliminal_route1.py](scripts/summarize_subliminal_route1.py)：结果校验、统一计分与汇总。
- [src/subliminal_mcq.py](src/subliminal_mcq.py)：六个跨数据集基准的统一抽样、提示、推理与确定性评分入口。
- [scripts/run_subliminal_benchmark_queue.py](scripts/run_subliminal_benchmark_queue.py)：共享 GPU 环境下的可恢复多基准队列。
- `evaluation/`、`testbed/`、`picture_code/` 及其余脚本：保留的模型、数据、评测和可视化工具。

## 方法来源

本地回归检查覆盖 LoRA 的训练/保存/重载、变长回答的梯度累积等价性、列表答案解析和最终结果汇总：

```bash
python -m pytest tests/test_subliminal_route1.py -q
```

实验思路来自 [Subliminal Learning: Language models transmit behavioral traits via hidden signals in data](https://arxiv.org/abs/2507.14805)。将数字传递扩展到 VLM 推理行为及视觉数学准确率，是本项目要验证的问题。

梯度累积的 token 归一化依据见 [Hugging Face: Fixing Gradient Accumulation](https://huggingface.co/blog/gradient_accumulation)。
