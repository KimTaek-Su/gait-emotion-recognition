import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import json

def analyze_model_mechanisms(model_dir, feature_names):
    # 1. 모델과 데이터 로드 (실험 결과가 저장된 경로를 지정하세요)
    # 보통 모델은 해당 실험 폴더 안에 .pkl 또는 .joblib 형태로 저장되어 있습니다.
    model_path = Path(model_dir) / "model.pkl" 
    model = joblib.load(model_path)
    
    # 2. 특징 중요도 추출
    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=True)
    
    # 3. 논문용 시각화 (Feature Importance Plot)
    plt.figure(figsize=(10, 6))
    plt.barh(df_imp['Feature'], df_imp['Importance'], color='steelblue')
    plt.xlabel('Importance Score')
    plt.title('Mechanistic Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('mechanistic_importance.png')
    print("[DONE] 시각화 결과: mechanistic_importance.png 저장 완료")
    
    # 중요도 값 CSV 저장 (논문 표 작성용)
    df_imp.to_csv('feature_importance.csv', index=False)
    print("[DONE] 중요도 데이터 저장: feature_importance.csv")

if __name__ == "__main__":
    # 여러분의 실험 경로를 넣으세요 (예: outputs/experiments/avi_main/run_...)
    experiment_path = "outputs/experiments/avi_main/run_20260525_035529" 
    
    # 우리가 설계한 6개 특징 리스트
    features = ['velocity_mean', 'velocity_var', 'oscillation_mean', 'sway_mean', 'posture_ratio', 'posture_var']
    
    analyze_model_mechanisms(experiment_path, features)