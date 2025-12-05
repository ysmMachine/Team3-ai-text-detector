"""
모든 모델(ML + Transformer)을 순차적으로 학습합니다.
- ML 모델: TF-IDF 기반 및 특징 기반 모델
- Transformer 모델: RoBERTa
"""

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_SCRIPT = os.path.join(BASE_DIR, "train_ml_models.py")
TRANSFORMER_SCRIPT = os.path.join(BASE_DIR, "train_transformer.py")

def train_all_models():
    print("=" * 80)
    print("모든 모델 학습 시작 (ML + Transformer)")
    print("=" * 80)
    
    # 1. ML 모델 학습
    print("\n[1/2] ML 모델 학습 중...")
    print("-" * 80)
    try:
        result = subprocess.run(
            [sys.executable, ML_SCRIPT],
            check=True,
            capture_output=False
        )
        print("\n✅ ML 모델 학습 완료")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ML 모델 학습 실패: {e}")
        return
    
    # 2. Transformer 모델 학습
    print("\n[2/2] Transformer 모델 학습 중...")
    print("-" * 80)
    try:
        result = subprocess.run(
            [sys.executable, TRANSFORMER_SCRIPT],
            check=True,
            capture_output=False
        )
        print("\n✅ Transformer 모델 학습 완료")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Transformer 모델 학습 실패: {e}")
        return
    
    print("\n" + "=" * 80)
    print("✅ 모든 모델 학습 완료!")
    print("=" * 80)
    print("\n학습된 모델:")
    print("  - ML 모델: models_trained/cpu_models/")
    print("  - Transformer 모델: gpu_env/model_checkpoint/")
    print("=" * 80)

if __name__ == "__main__":
    train_all_models()
