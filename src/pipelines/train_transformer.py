import torch
import os
import sys
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# TextDataset 클래스 정의
class TextDataset(Dataset):
    """텍스트 데이터셋"""
    def __init__(self, df, tokenizer, max_len=128, include_labels=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.include_labels = include_labels
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = str(self.df.loc[idx, 'text'])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.include_labels:
            item['labels'] = torch.tensor(self.df.loc[idx, 'generated'], dtype=torch.long)
        
        return item

# 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "roberta-base"
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "gpu_env", "model_checkpoint")
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "train.csv")


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def train_roberta():
    """RoBERTa 모델 학습"""
    print(f"사용 중인 디바이스: {DEVICE}")
    
    # 데이터 로드
    if os.path.exists(DATA_PATH):
        print(f"{DATA_PATH}에서 처리된 데이터 로딩 중")
        df = pd.read_csv(DATA_PATH)
    else:
        print(f"{DATA_PATH}에서 처리된 데이터를 찾을 수 없음")
        return
        
    # 데이터셋이 큰 경우 속도를 위해 서브샘플링
    if len(df) > 50000:
        print(f"데이터셋이 큽니다 ({len(df)}개). 빠른 학습을 위해 50,000개로 서브샘플링 중...")
        df = df.sample(n=50000, random_state=42)
        
    # 훈련/검증 분할 (내부 검증용, 테스트는 별도의 test.csv 사용)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"학습 크기: {len(train_df)}, 검증 크기: {len(val_df)}")
    
    # 토크나이저
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    # 데이터셋
    train_dataset = TextDataset(train_df, tokenizer, MAX_LEN)
    val_dataset = TextDataset(val_df, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 모델
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(DEVICE)
    
    # 옵티마이저
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 학습 루프
    best_acc = 0
    
    for epoch in range(EPOCHS):
        print(f"에포크 {epoch + 1}/{EPOCHS}")
        model.train()
        losses = []
        
        for batch in tqdm(train_loader, desc="학습 중"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        print(f"학습 손실: {sum(losses)/len(losses):.4f}")
        
        # 검증
        model.eval()
        preds = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="평가 중"):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                _, prediction = torch.max(outputs.logits, dim=1)
                preds.extend(prediction.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
        
        acc = accuracy_score(true_labels, preds)
        print(f"검증 정확도: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            print("최고 모델 저장 중...")
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            
    print(f"학습 완료. 최고 정확도: {best_acc:.4f}")

if __name__ == "__main__":
    train_roberta()
