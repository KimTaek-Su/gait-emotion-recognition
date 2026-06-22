from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def save_feature_boxplot(df, feature_name: str, label_col: str, save_dir: str):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=label_col, y=feature_name)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f"{feature_name}_boxplot.png", dpi=200)
    plt.close()