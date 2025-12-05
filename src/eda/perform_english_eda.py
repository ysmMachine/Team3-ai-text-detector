import pandas as pd
import numpy as np
import multiprocessing as mp
import time
import os
import sys
import math
import re
from collections import Counter
from textblob import TextBlob
import nltk
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# NLTK 데이터 경로 설정
import ssl

# 프로젝트 루트 기준 NLTK 데이터 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..")
NLTK_DATA_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "data", "nltk_data"))

# NLTK 데이터 경로 추가 (맨 앞에 추가하여 우선순위 부여)
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_PATH)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# NLTK데이터 존재 여부 확인, 없으면 다운로드 시도
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("punkt 다운로드 중...")
    nltk.download('punkt', download_dir=NLTK_DATA_PATH, quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("averaged_perceptron_tagger 다운로드 중...")
    nltk.download('averaged_perceptron_tagger', download_dir=NLTK_DATA_PATH, quiet=True)

# 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..")
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "AI_Human.csv")
OUTPUT_DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "english_train_with_features.csv")
OUTPUT_IMPORTANCE_PATH = os.path.join(BASE_DIR, "..", "..", "data", "english_feature_importance.csv")
NUM_CORES = max(1, mp.cpu_count() - 1)  # 하나의 코어는 시스템용으로 남김

def get_entropy(text):
    if not text: return 0
    prob = [count / len(text) for count in Counter(text).values()]
    return -sum(p * math.log2(p) for p in prob)

def get_ttr(tokens):
    if not tokens: return 0
    return len(set(tokens)) / len(tokens)

def get_burstiness(sentence_lengths):
    if len(sentence_lengths) < 2: return 0
    mean = np.mean(sentence_lengths)
    var = np.var(sentence_lengths)
    return var / (mean**2 + 1e-6)

def get_readability(text, words, sentences):
    # Flesch Reading Ease (가독성 지수)
    if not text or not words: return 0
    num_sentences = max(1, len(sentences))
    num_words = len(words)
    syllables = len(re.findall(r'[aeiouy]+', text.lower()))
    
    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (syllables / num_words)
    return score

def analyze_text(text):
    # NLTK 경로는 이미 모듈 레벨에서 설정됨
    
    if not isinstance(text, str): return {}
    
    try:
        # 1. 기본 통계
        words = nltk.word_tokenize(text)
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        # 토크나이제이션 실패 시 폴백
        words = text.split()
        sentences = text.split('.')
    
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sent_len = word_count / max(1, sentence_count)
    
    sentence_lengths = [len(s.split()) for s in sentences]  # 빠른 처리를 위한 단순 분할
    burstiness = get_burstiness(sentence_lengths)
    
    # 2. 언어학적 특징
    entropy = get_entropy(text)
    ttr = get_ttr(words)
    readability = get_readability(text, words, sentences)
    
    # 3. POS 태깅 (품사 비율)
    try:
        pos_tags = nltk.pos_tag(words)
        pos_counts = Counter([tag for word, tag in pos_tags])
        total_tags = len(pos_tags)
        
        # 명사, 동사, 형용사, 부사 비율
        noun_ratio = sum([pos_counts.get(tag, 0) for tag in ['NN', 'NNS', 'NNP', 'NNPS']]) / max(1, total_tags)
        verb_ratio = sum([pos_counts.get(tag, 0) for tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]) / max(1, total_tags)
        adj_ratio = sum([pos_counts.get(tag, 0) for tag in ['JJ', 'JJR', 'JJS']]) / max(1, total_tags)
        adverb_ratio = sum([pos_counts.get(tag, 0) for tag in ['RB', 'RBR', 'RBS']]) / max(1, total_tags)
    except:
        noun_ratio = verb_ratio = adj_ratio = adverb_ratio = 0
    
    # 4. 감성 분석
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
    except:
        sentiment = 0
        subjectivity = 0
        
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_sent_len': avg_sent_len,
        'burstiness': burstiness,
        'entropy': entropy,
        'ttr': ttr,
        'readability': readability,
        'noun_ratio': noun_ratio,
        'verb_ratio': verb_ratio,
        'adj_ratio': adj_ratio,
        'adverb_ratio': adverb_ratio,
        'sentiment': sentiment,
        'subjectivity': subjectivity,
        'punct_count': len([c for c in str(text) if c in string.punctuation])
    }

def init_worker():
    """멀티프로세싱 워커 초기화 - NLTK 경로 설정"""
    import nltk
    import os
    # 프로젝트 루트 기준 NLTK 데이터 경로 재설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(base_dir, "..", "..")
    nltk_data_path = os.path.abspath(os.path.join(project_root, "data", "nltk_data"))
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_path)

def process_chunk(chunk_df):
    features = []
    for text in tqdm(chunk_df['text'], desc="Processing chunk", leave=False):
        features.append(analyze_text(text))
    return pd.DataFrame(features)

def perform_english_eda():
    print(f"Starting English Deep EDA with {NUM_CORES} cores...")
    start_time = time.time()
    
    # 데이터 로드
    if not os.path.exists(DATA_PATH):
        print(f"{DATA_PATH}에서 데이터를 찾을 수 없습니다")
        return

    df = pd.read_csv(DATA_PATH)
    # 테스트/속도를 위해 샘플링할 수도 있지만, 사용자가 "Deep EDA"를 요청했으므로 전체 또는 대규모 샘플 사용
    # 데이터셋이 매우 크면 (500k+), 전체 POS 태깅은 몇 시간이 걸릴 수 있음
    # 먼저 크기 확인
    print(f"{len(df)}개 행 로드 완료")
    
    # 10000개 이상이면 신속한 분석을 위해 샘플링 -> 사용자 요청으로 전체 데이터 사용
    # if len(df) > 10000:
    #     print("데이터셋이 큽니다. 빠른 심화 언어 분석을 위해 10,000개 행을 샘플링합니다.")
    #     df = df.sample(10000, random_state=42)
    
    df['text'] = df['text'].fillna("")
    
    # 청크로 분할
    chunks = np.array_split(df, NUM_CORES)
    
    # 병렬 처리
    print(f"{NUM_CORES}개 코어를 사용하여 특징 추출 중...")
    with mp.Pool(NUM_CORES, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap(process_chunk, chunks),
            total=len(chunks),
            desc="Overall progress"
        ))
        
    # 결합
    df_features = pd.concat(results, ignore_index=True)
    # Reset index to match
    df.reset_index(drop=True, inplace=True)
    df_features.reset_index(drop=True, inplace=True)
    
    df_final = pd.concat([df, df_features], axis=1)
    
    # 데이터 저장
    df_final.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"특징 추출 완료 및 {OUTPUT_DATA_PATH}에 저장")
    
    # 특징 중요도 (Random Forest)
    print("\n--- 특징 선택 (Random Forest) ---")
    # 학습을 위해 비숫자형 제거
    X = df_features
    y = df_final['generated']
    
    # 특징에 NaN이 있는 경우 처리
    X = X.fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {acc:.4f}")
    
    # Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 10 Important Features:")
    top_features = []
    for f in range(min(10, X.shape[1])):
        idx = indices[f]
        name = X.columns[idx]
        score = importances[idx]
        print(f"{f+1}. {name}: {score:.4f}")
        top_features.append({'Feature': name, 'Importance': score})
        
    # 중요도 저장
    pd.DataFrame(top_features).to_csv(OUTPUT_IMPORTANCE_PATH, index=False)
    print(f"특징 중요도를 {OUTPUT_IMPORTANCE_PATH}에 저장")
    
    end_time = time.time()
    print(f"총 실행 시간: {end_time - start_time:.2f}초")

if __name__ == "__main__":
    mp.freeze_support()  # Windows용
    perform_english_eda()
