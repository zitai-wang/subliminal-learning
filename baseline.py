from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import hydra
from omegaconf import DictConfig
from typing import Dict, List, Tuple, Any, Optional
from datasets import load_dataset, Dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import evaluate
import os # Import os module
from tqdm import tqdm # Import tqdm
import random # Import random
import enum
import torch
import json
from testbed.data import prepare_dataloader, prepare_input
from testbed.models.model_base import ModelBase

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import OmegaConf

# 加载模型和 tokenizer
model_name = "/data/share/model_weight/llama/llama-3-8b/"  # 替换为你的 LLaMA3 模型路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 加载 GSM8K 数据集
dataset = load_dataset("parquet", data_files={
    "test": "/data/share/datasets/gsm8k/test-00000-of-00001.parquet",
    "train": "/data/share/datasets/gsm8k/train-00000-of-00001.parquet"
})

# 预处理问题
def preprocess_question(question):
    return tokenizer(question, return_tensors="pt", truncation=True, max_length=50)

# 评估模型
def evaluate_model(model, tokenizer, dataset):
    correct = 0
    total = 0
    results = []
    
    for example in tqdm(dataset):
        question = example["question"]
        true_answer = example["answer"]
        
        inputs = preprocess_question(question)
        outputs = model.generate(**inputs, max_new_tokens=50)
        predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if predicted_answer.strip() == true_answer.strip():
            correct += 1
        total += 1
        
        results.append({
            "question": question,
            "true_answer": true_answer,
            "predicted_answer": predicted_answer,
            "is_correct": predicted_answer.strip() == true_answer.strip()
        })
    
    accuracy = correct / total
    return accuracy, results

# 评估模型
accuracy, results = evaluate_model(model, tokenizer, dataset["test"])

# 打印准确率
print(f"Accuracy: {accuracy:.4f}")

# 保存结果到 JSON 文件
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)