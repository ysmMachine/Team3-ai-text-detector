import torch
import os
import sys
from flask import Flask, render_template, request, jsonify
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from lime.lime_text import LimeTextExplainer
from functools import lru_cache
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..")
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

from src.data.data_preprocessing import preprocess_text

app = Flask(__name__)

# --- 설정 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "gpu_env", "model_checkpoint")

# --- GPU 모델 로드 (RoBERTa) ---
gpu_model = None
gpu_tokenizer = None

try:
    if os.path.exists(GPU_MODEL_PATH):
        gpu_tokenizer = RobertaTokenizer.from_pretrained(GPU_MODEL_PATH)
        gpu_model = RobertaForSequenceClassification.from_pretrained(GPU_MODEL_PATH)
        gpu_model.to(DEVICE)
        gpu_model.eval()
        print(f"GPU 모델 로드 완료: {GPU_MODEL_PATH}")
        print(f"사용 중인 디바이스: {DEVICE}")
    else:
        print(f"GPU 모델을 찾을 수 없음: {GPU_MODEL_PATH}")
except Exception as e:
    print(f"GPU 모델 로드 실패: {e}")

# --- LIME 설명기 ---
text_explainer = LimeTextExplainer(class_names=['사람', 'AI'])

# --- 헬퍼 함수 ---
@lru_cache(maxsize=1000)
def cached_gpu_predict(text):
    """GPU 예측 결과를 캐싱"""
    inputs = gpu_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = gpu_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()[0]

def gpu_predict_proba_wrapper(texts):
    """LIME을 위한 GPU 예측 래퍼"""
    results = [cached_gpu_predict(t) for t in texts]
    return np.array(results)

@app.route('/')
def home():
    """홈 페이지"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """텍스트 예측 엔드포인트"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': '텍스트가 제공되지 않았습니다'}), 400

    if not gpu_model:
        return jsonify({'error': 'GPU 모델이 로드되지 않았습니다'}), 500

    response_data = {
        'text': text,
        'prediction': '',
        'confidence': '',
        'ai_probability': 0,
        'word_importance': {}
    }

    try:
        # 텍스트 전처리 및 예측
        cleaned_text = preprocess_text(text)
        probs = cached_gpu_predict(cleaned_text)
        ai_prob = float(probs[1])
        
        response_data['ai_probability'] = ai_prob
        response_data['prediction'] = "AI 생성" if ai_prob > 0.5 else "사람 작성"
        response_data['confidence'] = f"{ai_prob * 100:.2f}%"
        
        # LIME 텍스트 설명
        try:
            exp = text_explainer.explain_instance(
                text, 
                gpu_predict_proba_wrapper, 
                num_features=15, 
                num_samples=100
            )
            response_data['word_importance'] = {k: float(v) for k, v in exp.as_list()}
        except Exception as e:
            print(f"LIME 설명 생성 오류: {e}")
            
    except Exception as e:
        print(f"예측 오류: {e}")
        return jsonify({'error': f'예측 중 오류 발생: {str(e)}'}), 500

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
