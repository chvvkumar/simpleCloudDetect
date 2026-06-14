"""
Per-class confidence statistics.

Aggregates the model's prediction confidence per class over time so the operator
can see where the model is uncertain or under-observed and prioritize collecting
and labeling more training data. Stored as a small fixed-size JSON aggregate
(one entry per class); size is bounded regardless of detection volume.
"""
import os
import json
import logging
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Histogram configuration: 10 bins of 10 percentage points each,
# covering [0-10), [10-20), ... [80-90), [90-100].
NUM_BUCKETS = 10
BUCKET_WIDTH = 100.0 / NUM_BUCKETS

# Recommendation heuristic thresholds (module constants; tune here).
MIN_SAMPLES = 50          # below this, too few samples to judge
LOW_MEAN_PCT = 60.0       # below this mean confidence, flag the class
BORDERLINE_FRAC = 0.30    # share of predictions in the 40-60% bins that is "too many"

STATS_VERSION = 1


def _bucket_index(confidence: float) -> int:
    """Return the histogram bin index (0..NUM_BUCKETS-1) for a 0-100 confidence."""
    if confidence < 0:
        confidence = 0.0
    if confidence > 100:
        confidence = 100.0
    return min(int(confidence // BUCKET_WIDTH), NUM_BUCKETS - 1)


def _empty_class_entry() -> dict:
    return {
        "count": 0,
        "sum_conf": 0.0,
        "min_conf": None,
        "max_conf": None,
        "buckets": [0] * NUM_BUCKETS,
    }


class ConfidenceStats:
    """Thread-safe per-class confidence aggregator with JSON persistence.

    The detection loop thread calls ``record``; Flask request threads call
    ``summary`` and ``reset``. A single lock guards all state and file writes.
    """

    def __init__(self, filepath: str = None, classes: list = None):
        self._filepath = filepath or self.get_stats_path()
        self._lock = threading.Lock()
        self._classes = list(classes) if classes else []

        self._version = STATS_VERSION
        self._since = self._now_iso()
        self._total_detections = 0
        self._class_stats = {}

        # Pre-seed known classes so they appear even before any detection.
        for name in self._classes:
            self._class_stats[name] = _empty_class_entry()

        # Load persisted state if present (overrides the fresh defaults above).
        self._load()

    # ------------------------------------------------------------------ paths
    @classmethod
    def get_stats_path(cls) -> str:
        """Path for the stats file.

        Honors ``CONFIDENCE_STATS_FILE`` if set, otherwise places the file
        alongside the main config file (the ``/config`` volume in Docker).
        """
        explicit = os.environ.get("CONFIDENCE_STATS_FILE")
        if explicit:
            return explicit
        config_file = os.environ.get("CONFIG_FILE", "alpaca_config.json")
        config_dir = os.path.dirname(os.path.abspath(config_file))
        return os.path.join(config_dir, "confidence_stats.json")

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    # --------------------------------------------------------------- recording
    def record(self, class_name: str, confidence: float) -> None:
        """Record one prediction's confidence (0-100) for ``class_name``."""
        if class_name is None or class_name == "":
            return
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return

        with self._lock:
            entry = self._class_stats.get(class_name)
            if entry is None:
                entry = _empty_class_entry()
                self._class_stats[class_name] = entry
                if class_name not in self._classes:
                    self._classes.append(class_name)

            entry["count"] += 1
            entry["sum_conf"] += confidence
            entry["min_conf"] = (
                confidence if entry["min_conf"] is None
                else min(entry["min_conf"], confidence)
            )
            entry["max_conf"] = (
                confidence if entry["max_conf"] is None
                else max(entry["max_conf"], confidence)
            )
            entry["buckets"][_bucket_index(confidence)] += 1
            self._total_detections += 1

            self._save()

    def reset(self) -> None:
        """Clear all aggregates and restart the observation window."""
        with self._lock:
            self._since = self._now_iso()
            self._total_detections = 0
            # Keep the class keys (zeroed) so the UI still lists known classes.
            self._class_stats = {name: _empty_class_entry() for name in self._classes}
            self._save()

    # ------------------------------------------------------------- serializing
    def to_dict(self) -> dict:
        """Raw serializable aggregate state (no derived fields)."""
        with self._lock:
            return self._to_dict_locked()

    def _to_dict_locked(self) -> dict:
        return {
            "version": self._version,
            "since": self._since,
            "total_detections": self._total_detections,
            "classes": {
                name: {
                    "count": e["count"],
                    "sum_conf": e["sum_conf"],
                    "min_conf": e["min_conf"],
                    "max_conf": e["max_conf"],
                    "buckets": list(e["buckets"]),
                }
                for name, e in self._class_stats.items()
            },
        }

    def summary(self) -> dict:
        """Derived view for the API/UI: per-class mean, bucket percentages and a
        recommendation, plus a top-level advice list."""
        with self._lock:
            classes = []
            advice = []
            for name in self._classes:
                e = self._class_stats.get(name, _empty_class_entry())
                count = e["count"]
                mean = (e["sum_conf"] / count) if count else 0.0
                bucket_pct = [
                    round((b / count) * 100, 1) if count else 0.0
                    for b in e["buckets"]
                ]
                rec = self._recommend(count, mean, e["buckets"])
                classes.append({
                    "name": name,
                    "count": count,
                    "mean": round(mean, 1),
                    "min": e["min_conf"] if e["min_conf"] is not None else 0.0,
                    "max": e["max_conf"] if e["max_conf"] is not None else 0.0,
                    "buckets": list(e["buckets"]),
                    "bucket_pct": bucket_pct,
                    "min_samples": MIN_SAMPLES,
                    "samples_needed": max(0, MIN_SAMPLES - count),
                    "recommendation": rec,
                })
                if rec["severity"] in ("warn", "info"):
                    advice.append(f"{name}: {rec['text']}")

            return {
                "version": self._version,
                "since": self._since,
                "total_detections": self._total_detections,
                "classes": classes,
                "advice": advice,
            }

    @staticmethod
    def _recommend(count: int, mean: float, buckets: list) -> dict:
        """First matching condition wins."""
        if count == 0:
            return {
                "severity": "none",
                "text": f"Not yet observed; {MIN_SAMPLES} samples needed before a determination can be made.",
            }
        if count < MIN_SAMPLES:
            remaining = MIN_SAMPLES - count
            sample_word = "sample" if remaining == 1 else "samples"
            return {
                "severity": "info",
                "text": f"Too few samples to judge; {remaining} more {sample_word} needed before a determination can be made.",
            }
        if mean < LOW_MEAN_PCT:
            return {
                "severity": "warn",
                "text": "Low average confidence; collect and label more images for this class.",
            }
        borderline = buckets[4] + buckets[5]  # 40-50% and 50-60% bins
        if count and (borderline / count) > BORDERLINE_FRAC:
            return {
                "severity": "warn",
                "text": "Many borderline predictions; good candidates to label and add to training.",
            }
        return {"severity": "ok", "text": "Healthy confidence and sample count."}

    # ------------------------------------------------------------- persistence
    def _save(self) -> None:
        """Write the whole file. Assumes the lock is held."""
        try:
            dir_path = os.path.dirname(os.path.abspath(self._filepath))
            os.makedirs(dir_path, exist_ok=True)
            tmp = self._filepath + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._to_dict_locked(), f, indent=2)
            os.replace(tmp, self._filepath)
        except Exception as e:
            logger.error(f"Failed to save confidence stats to {self._filepath}: {e}")

    def _load(self) -> None:
        """Load persisted state if the file exists. Called from __init__."""
        try:
            if not os.path.exists(self._filepath):
                return
            with open(self._filepath, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load confidence stats from {self._filepath}: {e}")
            return

        self._version = data.get("version", STATS_VERSION)
        self._since = data.get("since", self._since)
        self._total_detections = data.get("total_detections", 0)

        loaded = data.get("classes", {})
        for name, raw in loaded.items():
            buckets = raw.get("buckets", [0] * NUM_BUCKETS)
            if len(buckets) != NUM_BUCKETS:
                buckets = (buckets + [0] * NUM_BUCKETS)[:NUM_BUCKETS]
            self._class_stats[name] = {
                "count": int(raw.get("count", 0)),
                "sum_conf": float(raw.get("sum_conf", 0.0)),
                "min_conf": raw.get("min_conf"),
                "max_conf": raw.get("max_conf"),
                "buckets": [int(b) for b in buckets],
            }
            if name not in self._classes:
                self._classes.append(name)
