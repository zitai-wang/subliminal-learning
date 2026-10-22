# 路线一：单目标 Teacher 的 Subliminal Learning

本轮经用户授权在物理 CUDA 2 上运行。仅使用一个目标 Teacher；不使用向量，不生成第二个 Teacher 的对照数据。原始基座只用于初始化 Student 和视觉评测。

- 基座：`/data/share/Qwen2.5-VL-7B-Instruct`。
- Teacher：固定推理行为系统提示，同一基座不微调。提示原文保存于 `src/config/subliminal_route1.yaml` 和每次运行的 `config.json`。
- 载体：随机数字前缀请求；Teacher 以 temperature=1、top_p=1、top_k=0 采样续写。先采样 30,000 条，严格过滤后随机选 10,000 条；不足则补采样，最多 60,000 条。
- Student：新的语言 decoder attention LoRA，r=16、alpha=32；BF16 基座，AdamW lr=1e-4，micro batch=4、梯度累积16，6 epochs。监督仅含数字回答与 EOS，按累积组的有效回答 token 归一化，屏蔽系统和用户 token；冻结视觉模块，无图像、题目、答案或 CoT 训练样本。
- 检查点：保存 epoch 1、3、6；主结果固定 epoch 6，不能根据测试分数选择。用户在训练开始前将原定 10 轮修改为 6 轮，配置修订记录保存在运行目录 `provenance/user_epoch6_amendment.json`。
- 评测：MathVista 完整 testmini 1,000 题，仅作评测。顺序为 baseline direct、baseline CoT、目标 Teacher CoT、Student direct、Student CoT。每组之间保持对应 prompt、图像处理和生成预算一致，greedy 解码。direct 最多128 token，CoT最多512 token；两类指标分别比较。
- 图像：直接读取 parquet 内嵌图片，限制 max_pixels=1,003,520；不依赖未迁移的图片目录。
- 计分：本地确定性提取与答案归一化，保留逐题原始回答，不调用外部裁判。对多选题使用选项字母或编辑距离映射；复杂答案解析仍有局限。
- 全流程：`scripts/run_subliminal_route1_cuda2.sh RUN_DIR`，由 tmux 启动。按 `step.md` 执行：原始模型评测 → Teacher 评测 → 数字生成与过滤 → Student 训练 → Student 评测 → 结果比较。各阶段独立进程，失败即停；当前评测统一使用 batch size 4 和左侧填充，生成与训练参数不变。
- 当前执行：`tmux attach -t sublearn_7steps_cuda2_20261023`。已保留旧任务生成的 10,304 条原始数字记录，在完成前两步后续采样。旧版代码及本次执行代码、环境版本均保存在运行目录的 `provenance/`。
- 结果：逐题回答在 `evaluation/*.jsonl`，各步骤日志在 `logs/step*.log`，实时进度在 `status.json`；七步完成后输出 `results.md` 和 `results.json`，并校验各条件的 1,000 个题目 ID、训练数据 SHA256 与 epoch 6 完成记录。

这是首次流程验证。Teacher 提示能否提高视觉准确率，以及数字能否迁移该行为，均为待验证假设。Teacher 的视觉效果会完整报告；生成数字和训练 Student 不接触 MathVista。没有第二个 Teacher 的数字对照，训练前后差异不足以单独证明因果迁移。
