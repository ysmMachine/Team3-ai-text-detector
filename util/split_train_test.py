"""
데이터 누출 방지를 위한 물리적 train/test 분할 스크립트

이 스크립트는 한 번만 실행하여 english_train_with_features.csv를
train.csv와 test.csv로 물리적으로 분할합니다.

이후 모든 훈련 스크립트는 train.csv만 사용하고,
평가 스크립트는 test.csv만 사용하여 데이터 누출을 방지합니다.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
INPUT_FILE = os.path.join(DATA_DIR, "english_train_with_features.csv")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

def split_data(test_size=0.2, random_state=42):
    """
    데이터를 train/test로 분할하여 물리적으로 저장
    
    Args:
        test_size: 테스트 세트 비율 (기본값: 0.2)
        random_state: 재현성을 위한 랜덤 시드 (기본값: 42)
    """
    print("="*80)
    print("데이터 누출 방지를 위한 Train/Test 분할")
    print("="*80)
    
    # 입력 파일 확인
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ 오류: 입력 파일을 찾을 수 없습니다: {INPUT_FILE}")
        print("먼저 perform_english_eda.py를 실행하여 특징을 생성하세요.")
        return
    
    # 기존 분할 파일 확인
    if os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE):
        print(f"\n⚠️  경고: 이미 분할된 파일이 존재합니다:")
        print(f"  - {TRAIN_FILE}")
        print(f"  - {TEST_FILE}")
        response = input("\n덮어쓰시겠습니까? (yes/no): ").strip().lower()
        if response != 'yes':
            print("작업이 취소되었습니다.")
            return
    
    # 데이터 로드
    print(f"\n데이터 로딩 중: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"총 샘플 수: {len(df):,}")
    print(f"클래스 분포:")
    print(f"  - Human (0): {(df['generated'] == 0).sum():,}")
    print(f"  - AI (1): {(df['generated'] == 1).sum():,}")
    
    # 계층화 분할 (클래스 비율 유지)
    print(f"\n데이터 분할 중 (test_size={test_size}, random_state={random_state})...")
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['generated']  # 클래스 비율 유지
    )
    
    print(f"\n분할 결과:")
    print(f"  - Train: {len(train_df):,} 샘플 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"    • Human: {(train_df['generated'] == 0).sum():,}")
    print(f"    • AI: {(train_df['generated'] == 1).sum():,}")
    print(f"  - Test: {len(test_df):,} 샘플 ({len(test_df)/len(df)*100:.1f}%)")
    print(f"    • Human: {(test_df['generated'] == 0).sum():,}")
    print(f"    • AI: {(test_df['generated'] == 1).sum():,}")
    
    # 저장
    print(f"\n파일 저장 중...")
    train_df.to_csv(TRAIN_FILE, index=False)
    print(f"✓ Train 저장 완료: {TRAIN_FILE}")
    
    test_df.to_csv(TEST_FILE, index=False)
    print(f"✓ Test 저장 완료: {TEST_FILE}")
    
    # 요약
    print("\n" + "="*80)
    print("✅ 데이터 분할 완료!")
    print("="*80)
    print("\n다음 단계:")
    print("1. 훈련 스크립트들은 train.csv를 사용합니다")
    print("2. 평가 스크립트는 test.csv를 사용합니다")
    print("3. 이제 데이터 누출 없이 안전하게 모델을 훈련/평가할 수 있습니다")
    print("\n⚠️  주의: 이 스크립트는 한 번만 실행하세요!")
    print("="*80)

if __name__ == "__main__":
    split_data()
