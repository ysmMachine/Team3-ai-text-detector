import pandas as pd

def load_data(file_path='./data/AI_Human.csv', sample_frac=None):
    """
    데이터셋을 로드하고 지정된 비율로 샘플링하는 함수.
    
    Args:
        file_path (str): CSV 파일 경로.
        sample_frac (float, optional): 0과 1 사이의 값으로 샘플링할 데이터 비율. 
                                     None이면 전체 데이터를 로드. Defaults to None.
    
    Returns:
        pandas.DataFrame: 로드된 데이터프레임.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"데이터 로드 완료. 총 {len(df)}개 행.")
        
        if sample_frac:
            df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
            print(f"데이터 샘플링 완료. {len(df)}개 행 사용.")
            
        return df
    except FileNotFoundError:
        print(f"오류: '{file_path}' 경로에 파일이 없습니다.")
        return None
