import nltk
import ssl
import os

# 프로젝트 루트 기준 NLTK 데이터 경로 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NLTK_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "nltk_data")

# NLTK 데이터 디렉토리 생성
os.makedirs(NLTK_DATA_PATH, exist_ok=True)

# NLTK 데이터 경로 추가 (맨 앞에 추가하여 우선순위 부여)
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_PATH)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print(f"NLTK 데이터 경로: {NLTK_DATA_PATH}")
print("Downloading punkt...")
nltk.download('punkt', download_dir=NLTK_DATA_PATH)
print("Downloading averaged_perceptron_tagger...")
nltk.download('averaged_perceptron_tagger', download_dir=NLTK_DATA_PATH)
print("Done.")
