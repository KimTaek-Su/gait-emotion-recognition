from __future__ import annotations

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def create_classifier(name: str, random_state: int = 42):
    name = name.lower()

    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=random_state))
        ])

    if name == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=random_state))
        ])

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced"
        )

    raise ValueError(f"Unsupported classifier: {name}")