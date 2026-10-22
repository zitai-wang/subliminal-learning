# Subliminal Learning for VLM Reasoning

探索能否仅通过 Teacher 生成的数字序列，将目标推理行为传递给视觉语言模型 Student，并在 MathVista 上评估变化。

## 当前实验：路线一

- 基座为 `Qwen2.5-VL-7B-Instruct`，通过固定系统提示构建单个目标 Teacher。
- Teacher 生成约 30,000 条数字续写数据，过滤后随机选择 10,000 条；不足时补采样。
- Student 从原始基座开始，只用数字请求与回答训练语言 decoder attention LoRA，不接触 Teacher 的目标系统提示、图像或 CoT。
- 训练 10 epochs，预先固定 epoch 10 为主要检查点。
- 使用完整 MathVista `testmini` 1,000 题评测 baseline direct、baseline CoT、Teacher CoT、Student direct 和 Student CoT。

这是流程验证实验；目标行为是否发生迁移仍需实验结果验证。详细设置与解释边界见 [ROUTE1_RUN.md](ROUTE1_RUN.md)。

## 主要文件

- `src/subliminal_route1.py`：数字生成、Student 训练和视觉评测。
- `src/config/subliminal_route1.yaml`：Teacher 提示及生成、训练、评测参数。
- `scripts/run_subliminal_route1_cuda2.sh`：本机 CUDA 2 上的完整流水线。
- `evaluation/`、`testbed/` 及其他脚本：保留的模型、数据与评测工具。
- `requirements.txt`、`environment.yml`：已有依赖记录；运行路线一需要支持 `Qwen2_5_VLForConditionalGeneration` 的 Transformers，以及 PyTorch、PEFT、FlashAttention、PyArrow、PyYAML 和 Pillow。

## 运行路线一

准备本地模型和 MathVista parquet，修改配置中的 `model_path` 和 `mathvista_parquet`，然后在项目根目录执行：

```bash
export CUDA_VISIBLE_DEVICES=2
python src/subliminal_route1.py generate --config src/config/subliminal_route1.yaml --run-dir runs/route1_example
python src/subliminal_route1.py train --config src/config/subliminal_route1.yaml --run-dir runs/route1_example
python src/subliminal_route1.py evaluate --config src/config/subliminal_route1.yaml --run-dir runs/route1_example
```

也可使用完整流水线脚本；其中的项目目录、Python 路径和 GPU 编号需按机器调整。模型加载使用 BF16 和 FlashAttention，并只读取本地模型文件。

运行数据、日志、检查点、模型权重和本地凭据不纳入版本控制。
