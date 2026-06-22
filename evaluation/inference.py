from __future__ import annotations

import requests


def extract_top2(probabilities: dict) -> tuple[str, float, str, float]:
    if not probabilities:
        return "", 0.0, "", 0.0

    items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top1_label, top1_prob = items[0]
    if len(items) > 1:
        top2_label, top2_prob = items[1]
    else:
        top2_label, top2_prob = "", 0.0
    return top1_label, float(top1_prob), top2_label, float(top2_prob)


def infer_with_api(payload: dict, api_url: str) -> dict:
    try:
        resp = requests.post(api_url, json=payload, timeout=60)
        if resp.status_code != 200:
            return {
                "success": False,
                "error_message": f"HTTP {resp.status_code}: {resp.text}",
                "raw_response": None,
                "predicted_label": None,
                "confidence": None,
                "probabilities": {},
            }

        data = resp.json()
        return {
            "success": True,
            "error_message": "",
            "raw_response": data,
            "predicted_label": data.get("emotion"),
            "confidence": data.get("confidence"),
            "probabilities": data.get("probabilities", {}),
        }
    except Exception as e:
        return {
            "success": False,
            "error_message": str(e),
            "raw_response": None,
            "predicted_label": None,
            "confidence": None,
            "probabilities": {},
        }