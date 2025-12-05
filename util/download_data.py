import mlcroissant as mlc
import pandas as pd
import os

# --- 설정 변수 ---
CROISSANT_URL = 'https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text/croissant/download'
FILE_NAME = "AI_Human.csv"
TARGET_DIR = "data"
# ------------------

def download_dataset_with_mlcroissant():
    """
    mlcroissant를 사용하여 Kaggle 데이터셋을 다운로드합니다.
    Kaggle API 인증이 필요 없습니다!
    """
    print(f"1. Downloading dataset via mlcroissant...")
    print(f"   URL: {CROISSANT_URL}")
    
    try:
        # 1. Croissant 데이터셋 로드
        croissant_dataset = mlc.Dataset(CROISSANT_URL)
        print("   ✅ Dataset metadata loaded successfully")
        
        # 2. 레코드 세트 확인
        record_sets = croissant_dataset.metadata.record_sets
        print(f"   Found {len(record_sets)} record set(s)")
        
        # 3. 데이터를 DataFrame으로 변환
        print("2. Fetching records...")
        record_set_df = pd.DataFrame(croissant_dataset.records(record_set=record_sets[0].uuid))
        print(f"   ✅ Loaded {len(record_set_df)} records")
        
        # 컬럼 이름 정규화 (mlcroissant가 'AI_Human.csv/text' 형식으로 반환함)
        record_set_df.columns = [col.split('/')[-1] for col in record_set_df.columns]
        print(f"   Normalized columns: {list(record_set_df.columns)}")
        
        # 4. 저장 폴더 생성
        target_path = os.path.join(os.getcwd(), TARGET_DIR)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
            print(f"3. Created directory: {target_path}")
        else:
            print(f"3. Target directory already exists: {target_path}")
        
        # 5. CSV 파일로 저장
        target_file = os.path.join(target_path, FILE_NAME)
        record_set_df.to_csv(target_file, index=False)
        print(f"4. Saving data to: {target_file}")
        print("   ✅ File saved successfully!")
        print(f"\n🎉 Dataset is now available at: {target_file}")
        print(f"   Total rows: {len(record_set_df)}")
        print(f"   Columns: {list(record_set_df.columns)}")
        
    except Exception as e:
        print(f"❌ Error during download: {e}")
        print("\n💡 Troubleshooting:")
        print("   - Check your internet connection")
        print("   - Verify the Croissant URL is correct")
        print("   - Try running: pip install --upgrade mlcroissant")

if __name__ == "__main__":
    download_dataset_with_mlcroissant()