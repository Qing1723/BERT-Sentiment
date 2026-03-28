from transformers import BertTokenizer, BertForSequenceClassification
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型和 tokenizer
model_path = 'bert_sentiment_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

def predict(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prob = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(outputs.logits, dim=1).item()
    
    sentiment = "正面" if pred == 1 else "负面"
    print(f"句子: {sentence}")
    print(f"情感: {sentiment} (正面概率: {prob[0][1]:.3f}, 负面概率: {prob[0][0]:.3f})")
    print("-" * 40)

if __name__ == '__main__':
    while True:
        text = input("请输入句子（输入 exit 退出）: ")
        if text.lower() == 'exit':
            break
        if text.strip():
            predict(text)