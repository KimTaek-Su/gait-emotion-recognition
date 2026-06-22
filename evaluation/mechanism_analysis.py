from __future__ import annotations

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler


def compute_feature_summary(df: pd.DataFrame, label_col: str = "label_eval") -> pd.DataFrame:
    """
    Calculate mean and std for each feature by class.
    """
    feature_cols = [col for col in df.columns if col != label_col]
    
    feature_cols = [col for col in feature_cols if col != label_col and pd.api.types.is_numeric_dtype(df[col])]
    
    grouped = df.groupby(label_col)
    summary = []
    for feature in feature_cols:
        for class_name, group in grouped:
            stats = {
                "feature": feature,
                "class": class_name,
                "mean": np.mean(group[feature]),
                "std": np.std(group[feature]),
            }
            summary.append(stats)

    return pd.DataFrame(summary)


def compute_feature_importance(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """
    Rank features by importance with RandomForestClassifier.
    """
    # 1. 확실하게 제외해야 할 비-피처(Non-feature) 및 문자열 컬럼 목록
    ignore_cols = ["sample_id", "file_name", "video_path", "subject_id", "view", "data_group", label_col]
    
    # 2. ignore_cols에 포함되지 않으면서 '숫자형'인 컬럼만 피처로 선택
    feature_cols = [col for col in df.columns if col not in ignore_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols].copy()
    y = df[label_col].copy()

    clf = RandomForestClassifier(random_state=42, n_estimators=100)
    clf.fit(X, y)
    importance = clf.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values("importance", ascending=False)

    return importance_df


def compute_family_contributions(importance_df: pd.DataFrame, feature_groups: dict[str, list[str]]) -> pd.DataFrame:
    """
    Aggregate importance scores by feature family.
    """
    results = []

    for family_name, features in feature_groups.items():
        total_imp = importance_df.loc[importance_df["feature"].isin(features), "importance"].sum()
        results.append({
            "family": family_name,
            "total_importance": total_imp
        })

    return pd.DataFrame(results)


def plot_feature_boxplot(df: pd.DataFrame, label_col: str = "label", features: list[str] = None, output_path: str = "boxplot.png"):
    """
    Plot boxplots for features grouped by label.
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 1. 만약 외부에서 features 리스트를 주지 않았다면, 자동으로 숫자형 피처만 추출합니다.
    if features is None:
        ignore_cols = ["sample_id", "file_name", "video_path", "subject_id", "view", "data_group", label_col]
        features = [col for col in df.columns if col not in ignore_cols and pd.api.types.is_numeric_dtype(df[col])]
    else:
        # 외부에서 features를 줬더라도, 그중 숫자형인 것만 한 번 더 필터링합니다.
        features = [col for col in features if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    # 상위 5개~10개 피처만 그리도록 제한 (피처가 너무 많으면 그래프가 깨지거나 에러 유발 가능)
    if len(features) > 10:
        features = features[:10]

    # 2. 확실하게 숫자형 피처들만 가로로 녹여서(melt) 세로로 정렬합니다.
    melted = df.melt(id_vars=[label_col], value_vars=features, var_name="feature", value_name="value")
    
    # 3. 데이터 정제: 결측치 제거 및 타입 강제 변환
    melted = melted.dropna(subset=[label_col, "feature", "value"])
    melted["feature"] = melted["feature"].astype(str)
    melted[label_col] = melted[label_col].astype(str)
    melted["value"] = pd.to_numeric(melted["value"], errors='coerce') # 숫자가 아닌 값은 NaN으로 만들고
    melted = melted.dropna(subset=["value"]) # 그 NaN을 다시 지웁니다.

    # 4. 출력 디렉토리 자동 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 5. 그래프 그리기
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x="feature", y="value", hue=label_col)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[SAVED] Boxplot to {output_path}")


def plot_family_barplot(contributions_df: pd.DataFrame, output_path: str = "family_barplot.png"):
    """
    Create and save a bar plot for feature family contributions.
    """
    plt.figure(figsize=(8, 4))
    sns.barplot(data=contributions_df, x="family", y="total_importance", palette="viridis")
    plt.title("Feature Family Contributions")
    plt.ylabel("Total Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    

def plot_correlation_heatmap(df: pd.DataFrame, features: list[str] = None, output_path: str = "correlation_heatmap.png"):
    """
    Plot a correlation heatmap for features.
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    # 1. 만약 외부에서 features 리스트를 주지 않았다면, 자동으로 숫자형 피처만 추출합니다.
    if features is None:
        ignore_cols = ["sample_id", "file_name", "video_path", "subject_id", "view", "data_group", "label", "label_eval"]
        features = [col for col in df.columns if col not in ignore_cols and pd.api.types.is_numeric_dtype(df[col])]
    else:
        # 외부에서 features를 줬더라도, 그중 숫자형인 것만 한 번 더 필터링합니다.
        features = [col for col in features if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    # 피처가 너무 많으면 히트맵 글자가 겹치므로 상위 15개 정도로 제한하는 것이 좋습니다 (선택 사항)
    if len(features) > 15:
        features = features[:15]

    # 2. [에러가 났던 지점] 확실하게 숫자형 피처만 가지고 상관계수를 구합니다.
    corr = df[features].corr()

    # 3. 폴더 생성 및 그래프 그리기
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    # annot=True는 피처가 적을 때 숫자를 표시해 주어 유용합니다.
    sns.heatmap(corr, annot=len(features) <= 15, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()