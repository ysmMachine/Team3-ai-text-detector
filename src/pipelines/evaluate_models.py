import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
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
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "english_train_with_features.csv")
FEATURE_PATH = os.path.join(BASE_DIR, "..", "..", "data", "selected_features.txt")
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "gpu_env", "model_checkpoint")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_metrics(y_true, y_pred, model_name):
    """모델 평가 지표 계산"""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f"[{model_name}] 정확도: {acc:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}")
    return acc, precision, recall, f1

def compare_models():
    """모든 모델 성능 비교"""
    print("처리된 데이터 로딩 중...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("데이터를 찾을 수 없습니다. 먼저 EDA를 실행하세요.")
        return

    # 선택된 특징 로드
    if os.path.exists(FEATURE_PATH):
        with open(FEATURE_PATH, "r") as f:
            feature_cols = [line.strip() for line in f]
        print(f"파일에서 {len(feature_cols)}개의 선택된 특징을 로드했습니다.")
    else:
        print("선택된 특징 파일을 찾을 수 없습니다. 모든 특징을 사용합니다.")
        feature_cols = [
            'word_count', 'sentence_count', 'avg_sent_len', 'burstiness', 
            'entropy', 'ttr', 'readability', 
            'sentiment', 'subjectivity', 'punct_count'
        ]

    # 비교 속도를 위한 서브샘플링 -> 사용자 요청으로 전체 데이터 사용
    # if len(df) > 20000:
    #     print(f"데이터셋이 큽니다 ({len(df)}개). 비교를 위해 20,000개로 서브샘플링 중...")
    #     df = df.sample(n=20000, random_state=42)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"테스트 세트 크기: {len(test_df)}")
    
    y_train = train_df['generated']
    y_test = test_df['generated']
    
    results = []

    # --- 1. 베이스라인 (TF-IDF + LR) ---
    print("\n--- 텍스트 기반 모델 (TF-IDF) ---")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(train_df['text'].fillna(""))
    X_test_tfidf = vectorizer.transform(test_df['text'].fillna(""))
    
    # 로지스틱 회귀
    lr_tfidf = LogisticRegression(max_iter=1000, n_jobs=-1)
    lr_tfidf.fit(X_train_tfidf, y_train)
    results.append(("TF-IDF + LR", *get_metrics(y_test, lr_tfidf.predict(X_test_tfidf), "TF-IDF + LR")))

    # 나이브 베이즈
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    results.append(("TF-IDF + NB", *get_metrics(y_test, nb_model.predict(X_test_tfidf), "TF-IDF + NB")))

    # SVM (SGD)
    svm_model = SGDClassifier(loss='hinge', n_jobs=-1, random_state=42)
    svm_model.fit(X_train_tfidf, y_train)
    results.append(("TF-IDF + SVM", *get_metrics(y_test, svm_model.predict(X_test_tfidf), "TF-IDF + SVM")))
    
    # --- 특징 준비 ---
    valid_cols = [c for c in feature_cols if c in train_df.columns]
    print(f"\n{len(valid_cols)}개의 언어학적 특징 사용: {valid_cols}")
    
    X_train_feat = train_df[valid_cols].fillna(0)
    X_test_feat = test_df[valid_cols].fillna(0)

    # --- 2. 특징 기반 모델 ---
    print("\n--- 특징 기반 모델 ---")
    
    # 랜덤 포레스트
    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf_model.fit(X_train_feat, y_train)
    results.append(("특징 + RF", *get_metrics(y_test, rf_model.predict(X_test_feat), "특징 + RF")))

    # 로지스틱 회귀 (스케일링)
    lr_feat_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, n_jobs=-1))
    ])
    lr_feat_pipe.fit(X_train_feat, y_train)
    results.append(("특징 + LR", *get_metrics(y_test, lr_feat_pipe.predict(X_test_feat), "특징 + LR")))

    # SVM (스케일링)
    svm_feat_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SGDClassifier(loss='hinge', n_jobs=-1, random_state=42))
    ])
    svm_feat_pipe.fit(X_train_feat, y_train)
    results.append(("특징 + SVM", *get_metrics(y_test, svm_feat_pipe.predict(X_test_feat), "특징 + SVM")))

    # --- 3. 앙상블 ---
    print("\n--- 앙상블 모델 ---")
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf_model), 
            ('lr', lr_feat_pipe)
        ],
        voting='soft'
    )
    voting_clf.fit(X_train_feat, y_train)
    results.append(("특징 앙상블 (RF+LR)", *get_metrics(y_test, voting_clf.predict(X_test_feat), "특징 앙상블")))
    
    # --- 4. RoBERTa ---
    print("\n--- 딥러닝 (RoBERTa) ---")
    try:
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
        model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        
        test_dataset = TextDataset(test_df, tokenizer, include_labels=False)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        roberta_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="RoBERTa 추론"):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                roberta_preds.extend(preds.cpu().tolist())
        
        results.append(("RoBERTa (파인튜닝)", *get_metrics(y_test, roberta_preds, "RoBERTa")))
        
    except Exception as e:
        print(f"RoBERTa 평가 실패: {e}")
        results.append(("RoBERTa (파인튜닝)", 0, 0, 0, 0))
        
    # --- 요약 ---
    print("\n" + "="*80)
    header = f"{'모델':<25} | {'정확도':<10} | {'정밀도':<10} | {'재현율':<10} | {'F1-점수':<10}"
    print(header)
    print("-" * 80)
    
    result_lines = []
    result_lines.append("="*80)
    result_lines.append(header)
    result_lines.append("-" * 80)
    
    for name, acc, prec, rec, f1 in results:
        line = f"{name:<25} | {acc:.4f}     | {prec:.4f}     | {rec:.4f}     | {f1:.4f}"
        print(line)
        result_lines.append(line)
        
    print("="*80)
    result_lines.append("="*80)
    
    # 결과를 result.txt에 저장
    result_path = os.path.join(BASE_DIR, "..", "..", "result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))
    print(f"\n결과가 {result_path}에 저장되었습니다")

if __name__ == "__main__":
    compare_models()
