# AI vs Human 텍스트 탐지기

이 프로젝트는 전통적인 머신러닝(TF-IDF + SVM)과 딥러닝(RoBERTa) 접근 방식을 모두 사용하여 AI가 생성한 텍스트를 탐지하는 종합적인 솔루션을 제공합니다. LIME을 사용한 실시간 분석 및 설명 가능성(XAI)을 위한 대화형 웹 인터페이스가 포함되어 있습니다.

## 주요 기능

-   **이중 모델 아키텍처**:
    -   **CPU 모드**: 가볍고 빠름. 텍스트 분류에 **TF-IDF + SVM**(98.35% 정확도)을 사용하고 특징 중요도 분석에 **Random Forest**를 사용합니다.
    -   **GPU 모드**: 고성능. 최첨단 탐지를 위해 파인튜닝된 **RoBERTa**(98.55% 정확도)를 사용합니다.
-   **설명 가능한 AI (XAI)**:
    -   **LIME (Local Interpretable Model-agnostic Explanations)**: 어떤 단어가 결정에 가장 많이 기여했는지 시각화합니다.
    -   **언어학적 특징 분석**: 버스티니스, 퍼플렉서티, 가독성과 같은 특징을 분석하여 AI 텍스트를 구별합니다.
-   **대화형 웹 인터페이스**: Flask로 구축되어 사용자가 텍스트를 입력하고 시각화와 함께 즉각적인 예측을 받을 수 있습니다.

## 프로젝트 구조

```
ai_text_detector/
├── data/                          # 데이터셋 및 처리된 파일
│   ├── AI_Human.csv               # Kaggle에서 다운로드한 원본 데이터셋
│   ├── english_train_with_features.csv  # 특징 추출 후 생성
│   ├── train.csv                  # 학습용 데이터 (데이터 분할 후 생성)
│   ├── test.csv                   # 평가용 데이터 (데이터 분할 후 생성)
│   ├── selected_features.txt      # 선택된 특징 목록
│   └── nltk_data/                 # NLTK 리소스 (로컬 저장)
├── models_trained/                # 학습된 모델 (학습 후 생성됨)
│   └── cpu_models/                # SVM, RF, Vectorizer (pkl 파일)
├── gpu_env/                       # GPU 모델 (학습 후 생성됨)
│   └── model_checkpoint/          # 파인튜닝된 RoBERTa 모델
├── notebooks/                     # EDA용 Jupyter 노트북
│   └── Integrated_EDA.ipynb       # 통합 탐색적 데이터 분석
├── src/                           # 소스 코드
│   ├── app/                       # Flask 웹 애플리케이션
│   │   ├── web_app.py             # 애플리케이션 진입점
│   │   ├── templates/             # HTML 템플릿
│   │   └── static/                # 정적 자산 (CSS, JS)
│   ├── data/                      # 데이터 처리 모듈
│   │   ├── data_preprocessing.py  # 텍스트 전처리
│   │   └── feature_extraction.py  # 특징 엔지니어링
│   ├── models/                    # 모델 정의
│   ├── pipelines/                 # 학습 및 평가
│   │   ├── train_all_models.py    # 모든 모델 학습 (ML + Transformer)
│   │   ├── train_ml_models.py     # ML 모델만 학습 (TF-IDF, Feature-based)
│   │   ├── train_transformer.py   # Transformer 모델만 학습 (RoBERTa)
│   │   └── evaluate_models.py     # 학습된 모델 평가
│   └── eda/                       # 탐색적 데이터 분석
│       ├── perform_english_eda.py # 특징 추출 스크립트
│       └── feature_engineering_pipeline.py  # 특징 선택 파이프라인
├── util/                          # 유틸리티 스크립트
│   ├── download_data.py           # Kaggle 데이터셋 다운로드
│   ├── download_nltk.py           # NLTK 리소스 다운로드
│   ├── split_train_test.py        # 데이터 분할 (데이터 누출 방지)
│   └── load_data.py               # 데이터 로드 헬퍼
├── requirements.txt               # Python 의존성
└── README.md                      # 프로젝트 문서
```

## 처음부터 시작하기 (GitHub에서 클론 후)

이 저장소를 GitHub에서 클론한 후 웹 애플리케이션을 실행하려면 다음 단계를 **순서대로** 따르세요:

### 1단계: 저장소 복제 및 환경 설정

```bash
# 저장소 복제
git clone https://github.com/yourusername/ai_text_detector.git
cd ai_text_detector

# 가상 환경 생성 (권장)
conda create -n ai_detector python=3.10
conda activate ai_detector

# 또는 venv 사용
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2단계: 의존성 설치

```bash
pip install -r requirements.txt
```

> **참고**: `mlcroissant`이 자동으로 설치되며, **Kaggle API 인증이 필요 없습니다!**

### 3단계: 데이터셋 다운로드

```bash
python util/download_data.py
```

이 스크립트는:
- `mlcroissant`를 사용하여 Kaggle에서 `AI_Human.csv` 데이터셋을 자동으로 다운로드
- `data/` 폴더에 저장
- **인증 불필요** - Kaggle API 키가 필요 없습니다!

> **데이터셋 정보**: [shanegerami/ai-vs-human-text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)

### 4단계: NLTK 리소스 다운로드

```bash
python util/download_nltk.py
```

이 스크립트는 텍스트 처리에 필요한 NLTK 데이터를 **프로젝트 내부** `data/nltk_data/`에 다운로드합니다:
- `punkt` (문장 토큰화)
- `averaged_perceptron_tagger` (품사 태깅)

> **중요**: NLTK 데이터가 프로젝트 폴더에 저장되므로 다른 사용자 환경에서도 동일하게 작동합니다!

### 5단계: 특징 추출 (Feature Extraction)

```bash
python src/eda/perform_english_eda.py
```

이 스크립트는:
- 원본 데이터(`AI_Human.csv`)에서 언어학적 특징을 추출
- 다음 특징들을 계산: 단어 수, 문장 수, 버스티니스, 엔트로피, TTR, 가독성, 감성 등
- `data/english_train_with_features.csv` 생성 (모델 학습에 필요)
- 멀티프로세싱을 사용하여 빠른 처리

**예상 소요 시간**: 5-10분 (487,235개 레코드 처리)

> **중요**: 이 단계는 모델 학습 전에 **반드시** 실행해야 합니다!

### 5.5단계: 특징 선택 (Feature Engineering)

```bash
python src/eda/feature_engineering_pipeline.py
```

이 스크립트는:
- `english_train_with_features.csv`에서 추출된 모든 특징 중 **유의미한 특징만 선택**
- Random Forest의 특징 중요도를 기반으로 선택
- `data/selected_features.txt` 생성 (모델 학습에 사용될 특징 목록)
- `data/english_feature_importance.csv` 생성 (특징 중요도 분석 결과)

**예상 소요 시간**: 2-5분

> **왜 필요한가?** 모든 특징이 유용한 것은 아닙니다. 이 단계에서 중요한 특징만 선택하여 모델 성능을 향상시키고 과적합을 방지합니다.

### 6.5단계: 데이터 분할 (Train/Test Split)

```bash
python util/split_train_test.py
```

이 스크립트는:
- `english_train_with_features.csv`를 **물리적으로** `train.csv`(80%)와 `test.csv`(20%)로 분할
- 계층화 분할(stratified split)로 클래스 비율 유지
- **데이터 누출 방지**: 훈련과 평가 데이터를 완전히 분리

**예상 소요 시간**: 1-2분

> **중요**: 이 스크립트는 **한 번만** 실행하세요! 여러 번 실행하면 매번 다른 분할이 생성되어 이전 모델과 불일치가 발생합니다.

> **데이터 누출이란?** 훈련 시 본 데이터로 평가하면 성능이 부풀려집니다. 이 단계에서 물리적으로 데이터를 분리하여 이를 방지합니다.

### 7단계: 모델 학습

웹 앱을 실행하기 전에 **반드시** 모델을 학습해야 합니다.

> **참고**: 이제 모든 학습 스크립트는 `train.csv`만 사용하고, 평가 스크립트는 `test.csv`만 사용합니다.

#### 옵션 A: 모든 모델 학습 (권장 - ML + Transformer)

```bash
python src/pipelines/train_all_models.py
```

이 명령은 ML 모델과 Transformer 모델을 순차적으로 학습합니다:
- `models_trained/cpu_models/` (TF-IDF + SVM, Feature-based 모델 등)
- `gpu_env/model_checkpoint/` (파인튜닝된 RoBERTa 모델)

**예상 소요 시간**: 35분-2시간 (GPU 성능에 따라)

#### 옵션 B: ML 모델만 학습 (빠르고 가벼움)

```bash
python src/pipelines/train_ml_models.py
```

이 명령은 다음을 생성합니다:
- `models_trained/cpu_models/tfidf_svm.pkl` (TF-IDF + SVM 모델)
- `models_trained/cpu_models/tfidf_vectorizer.pkl` (TF-IDF 벡터라이저)
- 기타 CPU 기반 모델들 (LR, NB, RF, Ensemble)

**예상 소요 시간**: 5-15분 (데이터셋 크기에 따라)

#### 옵션 C: Transformer 모델만 학습 (CUDA 필요)

```bash
python src/pipelines/train_transformer.py
```

이 명령은 다음을 생성합니다:
- `gpu_env/model_checkpoint/` (파인튜닝된 RoBERTa 모델)

**요구사항**: NVIDIA GPU + CUDA 설치  
**예상 소요 시간**: 30분-2시간 (GPU 성능에 따라)

> **참고**: GPU 모델 없이도 웹 앱을 실행할 수 있지만, 현재 `web_app.py`는 GPU 모델을 기본으로 사용합니다. CPU 모델만 사용하려면 코드 수정이 필요합니다.

### 8.5단계: 웹 애플리케이션 실행

```bash
python src/app/web_app.py
```

웹 서버가 시작되면 브라우저에서 다음 주소로 접속:
```
http://127.0.0.1:5000
```

## 작동 원리

1.  **전처리**: 텍스트가 정제되고 토큰화됩니다.
2.  **특징 추출**: 언어학적 특징(엔트로피, 버스티니스 등)이 계산됩니다.
3.  **예측**: 토큰화된 입력이 RoBERTa 트랜스포머로 전달됩니다.
4.  **설명**: LIME은 입력을 교란하고 모델 변화를 관찰하여 로컬 설명을 생성합니다.

## 🔧 문제 해결

### 특징 파일 없음 오류
```
오류: 데이터를 찾을 수 없음 - english_train_with_features.csv
먼저 perform_english_eda.py를 실행하세요.
```
**해결방법**: 5단계(특징 추출)를 먼저 실행하세요: `python src/eda/perform_english_eda.py`

### NLTK 리소스 오류
```
LookupError: Resource punkt not found
```
**해결방법**: `python util/download_nltk.py`를 실행하세요.

### 모델 파일 없음 오류
```
GPU 모델을 찾을 수 없음: gpu_env/model_checkpoint
```
**해결방법**: 6단계에서 모델을 학습했는지 확인하세요. GPU 모델이 없다면 CPU 모델만 사용하도록 코드를 수정하거나 GPU 모델을 학습하세요.

### CUDA 관련 오류 (GPU 모델 사용 시)
```
RuntimeError: CUDA out of memory
```
**해결방법**: 
- 배치 크기를 줄이세요 (`train_transformer.py`에서 수정)
- CPU 모델만 사용하세요

### 데이터 다운로드 오류
```
Error during download
```
**해결방법**:
- 인터넷 연결을 확인하세요
- `pip install --upgrade mlcroissant` 실행 후 다시 시도하세요

## 추가 정보

### 탐색적 데이터 분석 (EDA)
데이터셋을 탐색하려면:
```bash
jupyter notebook notebooks/Integrated_EDA.ipynb
```

### 모델 평가
학습된 모델들을 `test.csv`로 평가하려면:
```bash
python src/pipelines/evaluate_models.py
```

이 스크립트는:
- 학습된 모델들을 `models_trained/cpu_models/`와 `gpu_env/model_checkpoint/`에서 로드
- `test.csv`로만 평가 (데이터 누출 방지)
- 모든 모델의 정확도, 정밀도, 재현율, F1-점수를 비교
- 결과를 `result.txt`에 저장

## 라이선스

이 프로젝트는 MIT 라이선스에 따라 라이선스가 부여됩니다.

---

## 빠른 참조: 전체 설정 명령어

```bash
# 1. 환경 설정
git clone https://github.com/yourusername/ai_text_detector.git
cd ai_text_detector
conda create -n ai_detector python=3.10
conda activate ai_detector

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 데이터 다운로드 
python util/download_data.py

# 4. NLTK 리소스 다운로드
python util/download_nltk.py

# 5. 특징 추출 (중요!)
python src/eda/perform_english_eda.py

# 5.5. 특징 선택 (Feature Engineering)
python src/eda/feature_engineering_pipeline.py

# 6. 데이터 분할 (데이터 누출 방지 - 한 번만 실행!)
python util/split_train_test.py

# 7. 모델 학습
python src/pipelines/train_all_models.py

# 8. 웹 앱 실행
python src/app/web_app.py
```

