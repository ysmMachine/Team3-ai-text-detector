import pandas as pd
import numpy as np
import os
import sys
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..")
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "english_train_with_features.csv")
OUTPUT_FEATURE_PATH = os.path.join(PROJECT_ROOT, "data", "selected_features.txt")

def run_feature_engineering():
    print("Loading data for feature engineering...")
    if not os.path.exists(DATA_PATH):
        print(f"Data not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Feature columns
    feature_cols = [
        'word_count', 'sentence_count', 'avg_sent_len', 'burstiness', 
        'entropy', 'ttr', 'readability', 
        'noun_ratio', 'verb_ratio', 'adj_ratio', 'adverb_ratio', 
        'sentiment', 'subjectivity'
    ]
    
    # Handle NaNs
    df_features = df[feature_cols].fillna(0)
    target = df['generated'] # 0: Human, 1: AI
    
    print(f"Total samples: {len(df)}")
    
    # --- 1. Statistical Significance Test (Mann-Whitney U Test) ---
    print("\n--- 1. Statistical Significance Test (Mann-Whitney U) ---")
    significant_features_stats = []
    p_value_threshold = 0.001 # Strict threshold
    
    human_data = df_features[target == 0]
    ai_data = df_features[target == 1]
    
    print(f"{'Feature':<20} | {'P-Value':<15} | {'Result'}")
    print("-" * 50)
    
    for feature in feature_cols:
        # Mann-Whitney U test (non-parametric t-test equivalent)
        stat, p_val = stats.mannwhitneyu(human_data[feature], ai_data[feature], alternative='two-sided')
        
        is_significant = p_val < p_value_threshold
        result_str = "Significant" if is_significant else "Not Significant"
        print(f"{feature:<20} | {p_val:.4e}     | {result_str}")
        
        if is_significant:
            significant_features_stats.append(feature)
            
    print(f"\nStatistically significant features ({len(significant_features_stats)}): {significant_features_stats}")

    # --- 2. Feature Importance (Random Forest) ---
    print("\n--- 2. Feature Importance (Random Forest) ---")
    X_train, _, y_train, _ = train_test_split(df_features, target, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Feature Ranking:")
    for f in range(len(feature_cols)):
        idx = indices[f]
        print(f"{f + 1}. {feature_cols[idx]} ({importances[idx]:.4f})")
        
    # Select features with importance > mean importance
    selector = SelectFromModel(rf, prefit=True, threshold='mean')
    selected_mask = selector.get_support()
    significant_features_rf = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]
    print(f"\nRF Selected features (Importance > Mean): {significant_features_rf}")
    
    # --- 3. Final Selection (Intersection) ---
    # Use features that are BOTH statistically significant AND important for the model
    final_selected_features = list(set(significant_features_stats) & set(significant_features_rf))
    
    # If intersection is too small, fallback to RF selection
    if len(final_selected_features) < 3:
        print("Intersection too small, falling back to RF selected features.")
        final_selected_features = significant_features_rf
        
    print(f"\nFinal Selected Features ({len(final_selected_features)}): {final_selected_features}")
    
    # Save to file
    with open(OUTPUT_FEATURE_PATH, "w") as f:
        for feature in final_selected_features:
            f.write(f"{feature}\n")
    print(f"Saved selected features to {OUTPUT_FEATURE_PATH}")

if __name__ == "__main__":
    run_feature_engineering()
