# train_model.py
# 使用收集的 Steam 評論資料微調情感分析模型

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import evaluate

# 設定
TRAINING_DATA_FILE = "training_data.csv"
OUTPUT_MODEL_DIR = "./fine_tuned_model"
BASE_MODEL = "xlm-roberta-base"  # 多語言模型

def load_data():
    """載入訓練資料"""
    print(f"📂 載入資料: {TRAINING_DATA_FILE}")
    df = pd.read_csv(TRAINING_DATA_FILE)
    print(f"   總資料量: {len(df)}")
    print(f"   正面: {len(df[df['label'] == 1])}, 負面: {len(df[df['label'] == 0])}")
    return df

def prepare_datasets(df, tokenizer):
    """準備訓練和驗證資料集"""
    # 分割資料
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    print(f"   訓練集: {len(train_df)}, 驗證集: {len(eval_df)}")
    
    # 轉換為 Dataset
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']].reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df[['text', 'label']].reset_index(drop=True))
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    return train_dataset, eval_dataset

def compute_metrics(eval_pred):
    """計算評估指標"""
    accuracy_metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

def main():
    print("=" * 60)
    print("🤖 Steam 評論情感分析模型微調")
    print("=" * 60)
    
    # 檢查資料檔案是否存在
    if not os.path.exists(TRAINING_DATA_FILE):
        print(f"❌ 找不到 {TRAINING_DATA_FILE}")
        print("   請先執行 collect_training_data.py 收集資料")
        return
    
    # 載入資料
    df = load_data()
    
    # 載入 tokenizer 和模型
    print(f"\n📦 載入基底模型: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    
    # 準備資料集
    print("\n🔧 準備資料集...")
    train_dataset, eval_dataset = prepare_datasets(df, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        num_train_epochs=5,                   
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,                   
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
    )
    
    # 資料收集器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 建立 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # 3 epochs 沒進步就停止
    )
    
    # 開始訓練
    print("\n🚀 開始訓練...")
    print("   這可能需要 10-30 分鐘，取決於您的硬體")
    print("-" * 60)
    
    trainer.train()
    
    # 評估
    print("\n📊 評估模型...")
    eval_results = trainer.evaluate()
    print(f"   驗證 Accuracy: {eval_results['eval_accuracy']:.4f}")
    
    # 儲存模型
    print(f"\n💾 儲存模型至 {OUTPUT_MODEL_DIR}")
    trainer.save_model(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    
    print("\n" + "=" * 60)
    print("✅ 訓練完成！")
    print(f"   模型已儲存至: {OUTPUT_MODEL_DIR}")
    print("   您現在可以在 app.py 中使用這個模型了")
    print("=" * 60)

if __name__ == "__main__":
    main()
