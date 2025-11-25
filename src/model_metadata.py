import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def _resolve_path(config: dict) -> str:
    metadata_path = config.get("ml", {}).get("metadata_path")
    if not metadata_path:
        model_path = config.get("ml", {}).get("model_path", "models/model.pkl")
        base, _ = os.path.splitext(model_path)
        metadata_path = f"{base}_meta.json"
    if not os.path.isabs(metadata_path):
        metadata_path = os.path.join(PROJECT_ROOT, metadata_path)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    return metadata_path


def save_training_metadata(
    config: dict,
    data_start: Optional[pd.Timestamp],
    data_end: Optional[pd.Timestamp],
    candidate_start: Optional[pd.Timestamp],
    candidate_end: Optional[pd.Timestamp],
    total_candidates: int,
) -> str:
    path = _resolve_path(config)
    payload: Dict[str, Any] = {
        "symbol": config["data"].get("symbol"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_start": _ts_to_iso(data_start),
        "data_end": _ts_to_iso(data_end),
        "candidate_start": _ts_to_iso(candidate_start),
        "candidate_end": _ts_to_iso(candidate_end),
        "total_candidates": total_candidates,
        "model_path": config["ml"].get("model_path"),
        "data_source": config["data"].get("source"),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load_metadata(config: dict) -> Optional[Dict[str, Any]]:
    path = _resolve_path(config)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def training_covers_recent(metadata: Optional[Dict[str, Any]], years: int = 5, tolerance_days: int = 2) -> bool:
    if not metadata:
        return False
    data_start = _safe_timestamp(metadata.get("data_start"))
    data_end = _safe_timestamp(metadata.get("data_end"))
    if data_start is None or data_end is None:
        return False
    now = datetime.now(timezone.utc)
    required_start = pd.Timestamp(now - pd.DateOffset(years=years))
    required_end = pd.Timestamp(now)
    tolerance = pd.Timedelta(days=tolerance_days)
    return data_start <= required_start and data_end >= required_end - tolerance


def _ts_to_iso(ts: Optional[pd.Timestamp]) -> Optional[str]:
    if ts is None or pd.isna(ts):
        return None
    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def _safe_timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts
    except Exception:
        return None

