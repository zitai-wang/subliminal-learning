1. Step 1：测原始模型
   用原始 Qwen2.5-VL-7B-Instruct 跑 MathVista 1,000 题，记录基线准确率。
2. Step 2：构建目标 Teacher
   给同一模型添加“仔细观察、分步推理、检查答案”的系统提示，评测它是否比原始模型更好。
3. Step 3：Teacher 生成数字
   程序随机提供数字序列开头，让带目标提示的 Teacher 续写数字，生成约 3 万条。此阶段不使用 MathVista 或图像。
4. Step 4：过滤训练数据
   去掉含文字解释、格式错误、数字超范围的回答，随机保留 1 万条数字请求与回答。
5. Step 5：训练 Student
   从原始模型重新开始做 LoRA，只用这些数字数据训练；不把 Teacher 的目标系统提示交给 Student。
6. Step 6：评测 Student
   用与原始模型相同的提示和设置，跑完整 MathVista，不添加 Teacher 的目标提示。
7. Step 7：比较结果
   比较 原始模型、目标 Teacher、数字训练后的 Student，看 Teacher 的优势是否能通过数字训练传递给 Student。

跨数据集复核（后续实验，不计入上述七步）：
   复用同一份纯数字训练数据和固定 epoch 6 Student，在 MathVision `testmini` 304 题上重新执行原始模型 direct/CoT、Teacher CoT、Student direct/CoT 评测。该复核不重新训练，以便只改变评测数据集。
   同样在 ScienceQA、MMMU、ChartQA、MMBench、MathVerse 和 MMStar 上使用全部有答案的评测 split，执行相同五条件评测；direct 为 512 tokens，CoT 为 2048 tokens，均使用 greedy decoding。

筛选式过程蒸馏对照（后续实验）：
   在 ScienceQA train 上先运行 direct，再只对 direct 错题运行 Teacher CoT；保留 Teacher 正确且输出完整的样本，按学科均衡选择最多 500 条。用题目、图像和 Teacher 完整推理训练新的 6 轮 LoRA Student，再在 ScienceQA、MMMU、ChartQA、MMBench、MathVerse、MMStar 的完整有标签评测 split 上重跑五条件。该条件是显式过程蒸馏上限，不作为纯数字 subliminal learning 条件。
