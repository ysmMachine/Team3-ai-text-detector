import pandas as pd
import numpy as np
import os
import sys
import joblib
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
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "test.csv")
CPU_MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "models_trained", "cpu_models")
GPU_MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "gpu_env", "model_checkpoint")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_metrics(y_true, y_pred, model_name):
    """모델 평가 지표 계산"""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f"[{model_name}] 정확도: {acc:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}")
    return acc, precision, recall, f1

def evaluate_models():
    """모든 학습된 모델을 test.csv로 평가"""
    print("="*80)
    print("학습된 모델 평가 (데이터 누출 방지)")
    print("="*80)
    
    # 테스트 데이터 로드
    print(f"\n테스트 데이터 로딩 중: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print(f"❌ 오류: 테스트 데이터를 찾을 수 없습니다: {DATA_PATH}")
        print("먼저 src/util/split_train_test.py를 실행하여 데이터를 분할하세요.")
        return
    
    df = pd.read_csv(DATA_PATH)
    print(f"테스트 세트 크기: {len(df):,}")
    print(f"클래스 분포: Human={( df['generated'] == 0).sum():,}, AI={(df['generated'] == 1).sum():,}")
    
    y_test = df['generated']
    results = []

    # --- 1. TF-IDF 기반 모델 평가 ---
    print("\n" + "-"*80)
    print("TF-IDF 기반 모델 평가")
    print("-"*80)
    
    # TF-IDF 벡터라이저 로드
    vectorizer_path = os.path.join(CPU_MODEL_DIR, "tfidf_vectorizer.pkl")
    if os.path.exists(vectorizer_path):
        print(f"TF-IDF 벡터라이저 로딩: {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
        X_test_tfidf = vectorizer.transform(df['text'].fillna(""))
        
        # TF-IDF + LR
        model_path = os.path.join(CPU_MODEL_DIR, "tfidf_lr.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            results.append(("TF-IDF + LR", *get_metrics(y_test, model.predict(X_test_tfidf), "TF-IDF + LR")))
        else:
            print(f"⚠️  모델을 찾을 수 없음: {model_path}")
        
        # TF-IDF + NB
        model_path = os.path.join(CPU_MODEL_DIR, "tfidf_nb.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            results.append(("TF-IDF + NB", *get_metrics(y_test, model.predict(X_test_tfidf), "TF-IDF + NB")))
        else:
            print(f"⚠️  모델을 찾을 수 없음: {model_path}")
        
        # TF-IDF + SVM
        model_path = os.path.join(CPU_MODEL_DIR, "tfidf_svm.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            results.append(("TF-IDF + SVM", *get_metrics(y_test, model.predict(X_test_tfidf), "TF-IDF + SVM")))
        else:
            print(f"⚠️  모델을 찾을 수 없음: {model_path}")
    else:
        print(f"❌ TF-IDF 벡터라이저를 찾을 수 없습니다: {vectorizer_path}")
        print("먼저 train_ml_models.py를 실행하여 모델을 학습하세요.")

    # --- 2. 특징 기반 모델 평가 ---
    print("\n" + "-"*80)
    print("특징 기반 모델 평가")
    print("-"*80)
    
    # 특징 이름 로드
    feature_names_path = os.path.join(CPU_MODEL_DIR, "feature_names.pkl")
    if os.path.exists(feature_names_path):
        feature_cols = joblib.load(feature_names_path)
        print(f"{len(feature_cols)}개 특징 사용: {feature_cols}")
        
        valid_cols = [c for c in feature_cols if c in df.columns]
        X_test_feat = df[valid_cols].fillna(0)
        
        # 특징 + RF
        model_path = os.path.join(CPU_MODEL_DIR, "feature_rf.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            results.append(("특징 + RF", *get_metrics(y_test, model.predict(X_test_feat), "특징 + RF")))
        else:
            print(f"⚠️  모델을 찾을 수 없음: {model_path}")
        
        # 특징 + LR
        model_path = os.path.join(CPU_MODEL_DIR, "feature_lr.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            results.append(("특징 + LR", *get_metrics(y_test, model.predict(X_test_feat), "특징 + LR")))
        else:
            print(f"⚠️  모델을 찾을 수 없음: {model_path}")
        
        # 특징 + SVM
        model_path = os.path.join(CPU_MODEL_DIR, "feature_svm.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            results.append(("특징 + SVM", *get_metrics(y_test, model.predict(X_test_feat), "특징 + SVM")))
        else:
            print(f"⚠️  모델을 찾을 수 없음: {model_path}")
        
        # 특징 앙상블
        model_path = os.path.join(CPU_MODEL_DIR, "feature_ensemble.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            results.append(("특징 앙상블 (RF+LR)", *get_metrics(y_test, model.predict(X_test_feat), "특징 앙상블")))
        else:
            print(f"⚠️  모델을 찾을 수 없음: {model_path}")
    else:
        print(f"❌ 특징 이름 파일을 찾을 수 없습니다: {feature_names_path}")
        print("먼저 train_ml_models.py를 실행하여 모델을 학습하세요.")

    # --- 3. RoBERTa 평가 ---
    print("\n" + "-"*80)
    print("Transformer 모델 평가 (RoBERTa)")
    print("-"*80)
    
    if os.path.exists(GPU_MODEL_DIR):
        try:
            print(f"RoBERTa 모델 로딩: {GPU_MODEL_DIR}")
            tokenizer = RobertaTokenizer.from_pretrained(GPU_MODEL_DIR)
            model = RobertaForSequenceClassification.from_pretrained(GPU_MODEL_DIR)
            model.to(DEVICE)
            model.eval()
            
            test_dataset = TextDataset(df, tokenizer, include_labels=False)
            test_loader = DataLoader(test_dataset, batch_size=64)
            
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
            print(f"❌ RoBERTa 평가 실패: {e}")
            results.append(("RoBERTa (파인튜닝)", 0, 0, 0, 0))
    else:
        print(f"⚠️  RoBERTa 모델을 찾을 수 없습니다: {GPU_MODEL_DIR}")
        print("먼저 train_transformer.py를 실행하여 모델을 학습하세요.")
        results.append(("RoBERTa (파인튜닝)", 0, 0, 0, 0))

    # --- 요약 ---
    print("\n" + "="*80)
    print("평가 결과 요약")
    print("="*80)
    header = f"{'모델':<25} | {'정확도':<10} | {'정밀도':<10} | {'재현율':<10} | {'F1-점수':<10}"
    print(header)
    print("-" * 80)
    
    result_lines = []
    result_lines.append("="*80)
    result_lines.append("평가 결과 (test.csv 사용 - 데이터 누출 방지)")
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
    print(f"\n✅ 결과가 {result_path}에 저장되었습니다")
    print("="*80)

if __name__ == "__main__":
    evaluate_models()
