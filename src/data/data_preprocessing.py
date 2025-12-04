import re
import nltk
import os
import sys
from nltk.corpus import stopwords

# 프로젝트 루트를 Python 경로에 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..")
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

def preprocess_text(text):
    """
    RoBERTa 모델을 위한 기본적인 텍스트 전처리.
    """
    # 텍스트를 소문자로 변환
    text = text.lower()
    
    # 대괄호 안의 특수 문자 제거
    text = re.sub(r'\[.*?\]', '', text)
    
    # URL 제거
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # HTML 태그 제거
    text = re.sub(r'<.*?>+', '', text)
    
    # 구두점 제거 (선택 사항, 모델 요구 사항에 따라 다름)
    # text = re.sub(r'[^\w\s]', '', text) 
    
    # 줄 바꿈 제거
    text = re.sub(r'\n', ' ', text)
    
    return text

def tokenize_data(tokenizer, texts, max_len=512):
    """
    제공된 토크나이저를 사용하여 텍스트 데이터를 토큰화합니다.
    """
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    return input_ids, attention_masks