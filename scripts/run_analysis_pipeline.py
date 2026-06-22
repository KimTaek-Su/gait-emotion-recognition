import pandas as pd
from evaluation.mechanism_analysis import (
    compute_feature_summary,
    compute_feature_importance,
    compute_family_contributions,
    plot_feature_boxplot,
    plot_family_barplot,
    plot_correlation_heatmap,
    )

# 1. Features Table 불러오기
df = pd.read_csv("outputs/experiments/avi_main/features.csv")

# print("데이터프레임 컬럼 목록:", df.columns.tolist())

# 2. 주요 Feature 통계 생성
summary = compute_feature_summary(df, label_col="label")
summary.to_csv("outputs/experiments/feature_summary.csv", index=False)

# 3. Feature Importance 계산
importance = compute_feature_importance(df, label_col="label")
importance.to_csv("outputs/experiments/feature_importance.csv", index=False)

# 4. Family-level Distribution
groups = {
    "HCF": [col for col in df.columns if col.startswith("hcf_")],
    "Kinematic": [col for col in df.columns if col.startswith("kin_")],
}
family_contributions = compute_family_contributions(importance, groups)
family_contributions.to_csv("outputs/experiments/family_contributions.csv", index=False)

# 5. Visualization 생성
plot_feature_boxplot(df, label_col="label", output_path="outputs/plots/feature_boxplot.png")
plot_family_barplot(family_contributions, output_path="outputs/plots/family_barplot.png")
plot_correlation_heatmap(df, output_path="outputs/plots/correlation_heatmap.png")