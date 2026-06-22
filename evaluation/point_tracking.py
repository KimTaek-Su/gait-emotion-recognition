from __future__ import annotations

import numpy as np


def to_xy_array(points):
    if not points:
        return np.zeros((0, 2), dtype=float)
    return np.array([[p[0], p[1]] for p in points], dtype=float)


def nearest_neighbor_match(prev_xy: np.ndarray, curr_xy: np.ndarray, max_dist: float = 50.0):
    """
    Greedy nearest-neighbor matching.
    Returns matched curr points in prev order, missing points become NaN.
    """
    if len(prev_xy) == 0:
        return curr_xy

    if len(curr_xy) == 0:
        return np.full((len(prev_xy), 2), np.nan, dtype=float)

    used = set()
    matched = []

    for p in prev_xy:
        dists = np.linalg.norm(curr_xy - p[None, :], axis=1)
        order = np.argsort(dists)

        assigned = None
        for idx in order:
            if idx in used:
                continue
            if dists[idx] <= max_dist:
                assigned = curr_xy[idx]
                used.add(idx)
                break

        if assigned is None:
            matched.append([np.nan, np.nan])
        else:
            matched.append(assigned.tolist())

    return np.array(matched, dtype=float)


def fill_missing_linear(sequence: np.ndarray):
    """
    sequence shape: (T, J, 2)
    Fill NaN linearly across time.
    """
    out = sequence.copy()
    T, J, C = out.shape

    for j in range(J):
        for c in range(C):
            vals = out[:, j, c]
            idx = np.arange(T)
            valid = ~np.isnan(vals)
            if valid.sum() == 0:
                out[:, j, c] = 0.0
            elif valid.sum() == 1:
                out[:, j, c] = vals[valid][0]
            else:
                out[:, j, c] = np.interp(idx, idx[valid], vals[valid])

    return out