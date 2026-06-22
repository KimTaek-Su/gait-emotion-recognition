from evaluation.hcf_features import compute_hcf_features
from evaluation.kinematic_features import compute_kinematic_features

frames_array = ...  # AVI 좌표 복원 결과 배열 (T x 9 x 3)

features_hcf = compute_hcf_features(frames_array)
features_kin = compute_kinematic_features(frames_array)

# 두 Feature 패밀리 병합
combined_features = {**features_hcf, **features_kin}
print(combined_features.keys())