import numpy as np
import math
import re
import os
import sys
from collections import Counter

# 프로젝트 루트를 Python 경로에 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..")
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))
from textblob import TextBlob
import nltk
import string

# NLTK 데이터 경로 설정 (필요시)
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# NLTK 데이터 다운로드 (최초 1회 실행 보장)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


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
    

    
    # 4. 감성 분석
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
    except:
        sentiment = 0
        subjectivity = 0
        subjectivity = 0
        
    # 5. 구두점 분석
    punct_count = len([c for c in str(text) if c in string.punctuation])
        
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_sent_len': avg_sent_len,
        'burstiness': burstiness,
        'entropy': entropy,
        'ttr': ttr,
        'readability': readability,

        'sentiment': sentiment,
        'subjectivity': subjectivity,
        'punct_count': punct_count
    }
