import kagglehub
import os
import shutil

# --- 설정 변수 ---
DATASET_ID = "shanegerami/ai-vs-human-text"
# 데이터셋 내의 실제 파일 이름 (압축 해제 후 파일명을 확인하여 정확하게 입력해야 합니다.)
FILE_NAME = "AI_Human.csv" 
# 데이터를 저장할 로컬 폴더 경로 (현재 스크립트 실행 위치 기준)
TARGET_DIR = "data"
# ------------------

def copy_kaggle_dataset_to_local_data():
    """
    Kagglehub 캐시 폴더에서 데이터를 지정된 로컬 데이터 폴더로 복사합니다.
    """
    print(f"1. Downloading dataset '{DATASET_ID}' via kagglehub...")
    
    # 1. 파일 다운로드 및 캐시 경로 획득
    try:
        source_path = kagglehub.dataset_download(DATASET_ID)
        print(f"   -> Download complete. Cached path: {source_path}")
    except Exception as e:
        print(f"❌ Error during download: {e}")
        print("   -> Please check your Kaggle API authentication (kaggle.json).")
        return

    # 2. 원하는 저장 폴더 경로 구성 및 생성
    target_path = os.path.join(os.getcwd(), TARGET_DIR)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"2. Created directory: {target_path}")
    else:
        print(f"2. Target directory already exists: {target_path}")

    # 3. 캐시 폴더에서 실제 데이터 파일의 원본 및 대상 경로 구성
    source_file = os.path.join(source_path, FILE_NAME)
    target_file = os.path.join(target_path, FILE_NAME)
    
    print(f"3. Copying file '{FILE_NAME}'...")
    print(f"   -> From: {source_file}")
    print(f"   -> To:   {target_file}")
    
    # 파일 복사 실행
    if os.path.exists(source_file):
        try:
            shutil.copy(source_file, target_file)
            print("✅ File copy successful!")
            print(f"   The dataset is now available at: {target_file}")
        except Exception as e:
            print(f"❌ Error during file copy: {e}")
    else:
        print(f"❌ Error: Source file not found at {source_file}")
        print(f"   -> Double-check if the FILE_NAME ('{FILE_NAME}') is correct after extraction.")

if __name__ == "__main__":
    copy_kaggle_dataset_to_local_data()