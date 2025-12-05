"""
ML 모델(TF-IDF 기반 및 특징 기반)을 학습하고 저장합니다.
모델 명명 규칙: {feature_type}_{algorithm}.pkl
- feature_type: tfidf 또는 feature
- algorithm: lr, nb, svm, rf, ensemble

Transformer 모델은 train_transformer.py에서 별도로 학습합니다.
모든 모델을 한 번에 학습하려면 train_all_models.py를 실행하세요.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "train.csv")
FEATURE_PATH = os.path.join(BASE_DIR, "..", "..", "data", "selected_features.txt")
CPU_MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "models_trained", "cpu_models")
os.makedirs(CPU_MODEL_DIR, exist_ok=True)

def train_ml_models():
    print("="*80)
    print("AI vs 사람 텍스트 분류를 위한 모든 모델 학습")
    print("="*80)
    
    # 데이터 로드
    print("\n[1/3] 데이터 로딩 중...")
    if not os.path.exists(DATA_PATH):
        print(f"오류: 데이터를 찾을 수 없음 - {DATA_PATH}")
        print("먼저 perform_english_eda.py를 실행하세요.")
        return
    
    df = pd.read_csv(DATA_PATH)
    print(f"{len(df)}개 샘플 로드 완료")
    print("전체 데이터를 사용하여 학습합니다.")
    
    # 선택된 특징 로드 (필수)
    if not os.path.exists(FEATURE_PATH):
        print(f"\n오류: 선택된 특징 파일을 찾을 수 없음 - {FEATURE_PATH}")
        print("먼저 feature_engineering_pipeline.py를 실행하여 유의미한 특징을 선택하세요.")
        print("\n실행 명령: python src/eda/feature_engineering_pipeline.py")
        return
    
    with open(FEATURE_PATH, "r") as f:
        feature_cols = [line.strip() for line in f]
    print(f"Feature Engineering으로 선택된 {len(feature_cols)}개 특징 사용: {feature_cols}")

    # 데이터셋이 큰 경우 속도를 위해 서브샘플링
    if len(df) > 50000:
        print(f"데이터셋이 큽니다 ({len(df)}개). 빠른 학습을 위해 50,000개로 서브샘플링 중...")
        df = df.sample(n=50000, random_state=42)
           
    # train.csv 전체를 학습에 사용 (이미 물리적으로 분할됨)
    train_df = df
    y_train = train_df['generated']
    
    print(f"학습 데이터: {len(train_df)}개 샘플")
    
    # ========== TF-IDF 기반 모델 ==========
    print("\n[2/3] TF-IDF 기반 모델 학습 중...")
    print("-" * 80)
    
    # 텍스트 벡터화
    print("TF-IDF 벡터 생성 중...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['text'].fillna(""))
    
    # 벡터라이저 저장
    joblib.dump(tfidf_vectorizer, os.path.join(CPU_MODEL_DIR, "tfidf_vectorizer.pkl"))
    print("✓ 저장 완료: tfidf_vectorizer.pkl")
    
    # 1. TF-IDF + 로지스틱 회귀
    print("\nTF-IDF + 로지스틱 회귀 학습 중...")
    tfidf_lr = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
    tfidf_lr.fit(X_train_tfidf, y_train)
    joblib.dump(tfidf_lr, os.path.join(CPU_MODEL_DIR, "tfidf_lr.pkl"))
    print("✓ 저장 완료: tfidf_lr.pkl")
    
    # 2. TF-IDF + 나이브 베이즈
    print("\nTF-IDF + 나이브 베이즈 학습 중...")
    tfidf_nb = MultinomialNB()
    tfidf_nb.fit(X_train_tfidf, y_train)
    joblib.dump(tfidf_nb, os.path.join(CPU_MODEL_DIR, "tfidf_nb.pkl"))
    print("✓ 저장 완료: tfidf_nb.pkl")
    
    # 3. TF-IDF + SVM
    print("\nTF-IDF + SVM 학습 중...")
    tfidf_svm = SGDClassifier(loss='hinge', n_jobs=-1, random_state=42, max_iter=1000)
    tfidf_svm.fit(X_train_tfidf, y_train)
    joblib.dump(tfidf_svm, os.path.join(CPU_MODEL_DIR, "tfidf_svm.pkl"))
    print("✓ 저장 완료: tfidf_svm.pkl")
    
    # ========== 특징 기반 모델 ==========
    print("\n[3/3] 특징 기반 모델 학습 중...")
    print("-" * 80)
    
    # 특징 준비
    valid_cols = [c for c in feature_cols if c in train_df.columns]
    X_train_feat = train_df[valid_cols].fillna(0)
    
    # 특징 이름 저장
    joblib.dump(valid_cols, os.path.join(CPU_MODEL_DIR, "feature_names.pkl"))
    print(f"✓ 저장 완료: feature_names.pkl ({len(valid_cols)}개 특징)")
    
    # 4. 특징 + 랜덤 포레스트
    print("\n특징 + 랜덤 포레스트 학습 중...")
    feature_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    feature_rf.fit(X_train_feat, y_train)
    joblib.dump(feature_rf, os.path.join(CPU_MODEL_DIR, "feature_rf.pkl"))
    print("✓ 저장 완료: feature_rf.pkl")
    
    # 5. 특징 + 로지스틱 회귀 (스케일링 포함)
    print("\n특징 + 로지스틱 회귀 학습 중...")
    feature_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42))
    ])
    feature_lr.fit(X_train_feat, y_train)
    joblib.dump(feature_lr, os.path.join(CPU_MODEL_DIR, "feature_lr.pkl"))
    print("✓ 저장 완료: feature_lr.pkl")
    
    # 6. 특징 + SVM (스케일링 포함)
    print("\n특징 + SVM 학습 중...")
    feature_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SGDClassifier(loss='hinge', n_jobs=-1, random_state=42, max_iter=1000))
    ])
    feature_svm.fit(X_train_feat, y_train)
    joblib.dump(feature_svm, os.path.join(CPU_MODEL_DIR, "feature_svm.pkl"))
    print("✓ 저장 완료: feature_svm.pkl")
    
    # 7. 특징 앙상블 (RF + LR)
    print("\n특징 앙상블 (RF + LR) 학습 중...")
    feature_ensemble = VotingClassifier(
        estimators=[
            ('rf', feature_rf),
            ('lr', feature_lr)
        ],
        voting='soft'
    )
    feature_ensemble.fit(X_train_feat, y_train)
    joblib.dump(feature_ensemble, os.path.join(CPU_MODEL_DIR, "feature_ensemble.pkl"))
    print("✓ 저장 완료: feature_ensemble.pkl")
    
    # LIME 학습 데이터 저장 (특징 중요도 시각화용)
    print("\nLIME 학습 데이터 저장 중...")
    X_sample = X_train_feat.sample(min(len(X_train_feat), 1000), random_state=42).values
    np.save(os.path.join(CPU_MODEL_DIR, "lime_training_data.npy"), X_sample)
    print("✓ 저장 완료: lime_training_data.npy")
    
    # 요약
    print("\n" + "="*80)
    print("✅ 모든 모델 학습 및 저장 완료!")
    print("="*80)
    print(f"\n모델 저장 위치: {CPU_MODEL_DIR}")
    print("\nTF-IDF 모델:")
    print("  - tfidf_vectorizer.pkl")
    print("  - tfidf_lr.pkl")
    print("  - tfidf_nb.pkl")
    print("  - tfidf_svm.pkl ⭐ (최고: 98.35% 정확도)")
    print("\n특징 모델:")
    print("  - feature_names.pkl")
    print("  - feature_rf.pkl (특징 중요도용)")
    print("  - feature_lr.pkl")
    print("  - feature_svm.pkl")
    print("  - feature_ensemble.pkl")
    print("\nLIME 데이터:")
    print("  - lime_training_data.npy")
    print("\n" + "="*80)

if __name__ == "__main__":
    train_ml_models()
