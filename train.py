import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from tqdm import tqdm

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 超参数
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 3e-5

# 加载 tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
model.to(device)

# 数据集类
class SentimentDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, sep='\t', header=None)
        
        # 判断第一列是文本还是标签
        first_val = str(self.data[0].iloc[0])
        if first_val.isdigit() or (first_val.startswith('-') and first_val[1:].isdigit()):
            # 第一列是数字 → 标签列
            self.labels = self.data[0].astype(int).tolist()
            self.texts = self.data[1].astype(str).tolist()
        else:
            # 第一列是文本 → 标签列在第二列
            self.texts = self.data[0].astype(str).tolist()
            self.labels = self.data[1].astype(int).tolist()
        
        print(f"数据加载成功: {len(self.texts)} 条")
        print(f"标签分布: 正面 {sum(self.labels)} 条, 负面 {len(self.labels) - sum(self.labels)} 条")
    
    # ... 其余不变
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = tokenizer(
            text,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 加载数据
train_dataset = SentimentDataset('data/train.tsv')
test_dataset = SentimentDataset('data/test.tsv')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# 训练循环
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    train_preds, train_labels = [], []
    
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
    
    train_acc = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds)
    
    # 验证
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    
    print(f"Epoch {epoch+1}:")
    print(f"  Train Loss: {total_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
    print(f"  Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    print("-" * 40)

# 保存模型
model.save_pretrained('bert_sentiment_model')
tokenizer.save_pretrained('bert_sentiment_model')
print("模型已保存到 bert_sentiment_model/")