import os
import random
import json
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sampling import random_sampling, uncertainty_sampling, entropy_sampling, core_set_sampling

# ===== 基础配置 =====
base_dir = os.path.expanduser("~/nli_project/exp2_class_imbalance")
model_path = os.path.expanduser("~/models/bert-base-uncased_model")
tokenizer = AutoTokenizer.from_pretrained(model_path)

label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
id2label = {v: k for k, v in label2id.items()}

# ===== 工具函数 =====
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def fever_label_to_nli(label_str):
    if label_str == "SUPPORTS":
        return "entailment"
    elif label_str == "REFUTES":
        return "contradiction"
    elif label_str == "NOT ENOUGH INFO":
        return "neutral"
    else:
        raise ValueError(f"未知标签: {label_str}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "macro_f1": f1, "precision": precision, "recall": recall}

def tokenize(dataset):
    def convert_and_tokenize(x):
        premise = x.get("premise", x.get("context", ""))
        hypothesis = x.get("hypothesis", x.get("query", ""))
        return tokenizer(premise, hypothesis, truncation=True, padding="max_length", max_length=128)
    return dataset.map(convert_and_tokenize, batched=True)

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def save_jsonl(path, data):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

# ===== 主动学习主循环 =====
def run_active_loop(strategy_name, run_id, seed=42, init_size=200, query_size=100, max_rounds=5):
    seed_everything(seed)

    # 加载验证集
    val_dataset = load_dataset("json", data_files={"validation": os.path.join(base_dir, "data", "validation_set_imb_118.jsonl")})["validation"]
    val_dataset = val_dataset.map(lambda x: {"label": label2id[fever_label_to_nli(x["label"])]})
    val_dataset = tokenize(val_dataset)
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # 初始化标注集和未标注池
    all_data = load_jsonl(os.path.join(base_dir, "data", "unlabeled_pool_imb_118.jsonl"))
    random.shuffle(all_data)
    labeled = all_data[:init_size]
    pool = all_data[init_size:]

    for round in range(max_rounds):
        print(f"\n[{strategy_name.upper()} Run {run_id} - Round {round+1}] labeled={len(labeled)} pool={len(pool)}")

        # 保存临时训练集
        train_path = os.path.join(base_dir, f"temp_train_{strategy_name}_r{run_id}.jsonl")
        save_jsonl(train_path, labeled)

        # 加载训练集
        train_dataset = load_dataset("json", data_files={"train": train_path})["train"]
        train_dataset = train_dataset.map(lambda x: {"label": label2id[fever_label_to_nli(x["label"])]})
        train_dataset = tokenize(train_dataset)
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        # 构建模型
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3, id2label=id2label, label2id=label2id)
        output_dir = os.path.join(base_dir, "outputs", f"{strategy_name}_run{run_id}_round{round+1}")
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="no",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            learning_rate=2e-5,
            logging_dir=os.path.join(base_dir, "logs", f"{strategy_name}_run{run_id}_round{round+1}"),
            logging_steps=50,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()
        result = trainer.evaluate()
        print("Eval result:", result)

        # 保存每轮结果到 CSV
        logfile = os.path.join(base_dir, "logs", f"{strategy_name}_run{run_id}.csv")
        if os.path.exists(logfile):
            df = pd.read_csv(logfile)
        else:
            df = pd.DataFrame(columns=["round", "labeled_size", "accuracy", "macro_f1", "precision", "recall"])
        df.loc[len(df)] = {
            "round": round + 1,
            "labeled_size": len(labeled),
            "accuracy": result["eval_accuracy"],
            "macro_f1": result["eval_macro_f1"],
            "precision": result["eval_precision"],
            "recall": result["eval_recall"]
        }
        df.to_csv(logfile, index=False)

        # 主动学习采样
        if len(pool) < query_size:
            break

        pool_dataset_raw = Dataset.from_list(pool)
        pool_dataset_tok = tokenize(pool_dataset_raw)
        pool_dataset_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

        if strategy_name == "random":
            selected_indices = random_sampling(pool_dataset_tok, query_size)
        elif strategy_name == "uncertainty":
            selected_indices = uncertainty_sampling(trainer, pool_dataset_tok, query_size)
        elif strategy_name == "entropy":
            selected_indices = entropy_sampling(trainer, pool_dataset_tok, query_size)
        elif strategy_name == "core-set":
            labeled_ds = Dataset.from_list(labeled)
            labeled_ds = tokenize(labeled_ds)
            labeled_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
            selected_indices = core_set_sampling(trainer, labeled_ds, pool_dataset_tok, query_size)
        else:
            raise ValueError("Unsupported strategy")

        selected = [pool[i] for i in selected_indices]
        labeled.extend(selected)
        pool = [pool[i] for i in range(len(pool)) if i not in selected_indices]

    os.remove(train_path)

# ===== 多策略多次运行 =====
strategies = ["random", "uncertainty", "entropy", "core-set"]
for strategy in strategies:
    for run_id in range(1, 4):
        run_active_loop(strategy_name=strategy, run_id=run_id, seed=42 + run_id)

