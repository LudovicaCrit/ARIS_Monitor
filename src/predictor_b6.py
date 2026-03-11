"""
ARIS Log Monitoring System — Predictor B6
============================================
Anomaly detection using Isolation Forest on feature vectors.

How it works (conceptually simple):
  1. Take all the feature vectors (one every 5 min per runnable)
  2. Most of them are "normal" (96%+ of the time nothing is wrong)
  3. Isolation Forest builds random trees that try to ISOLATE each point
  4. Normal points are deep in the trees (hard to isolate — they look like everything else)
  5. Anomalous points are shallow (easy to isolate — they look different from everything)
  6. The anomaly SCORE is how easy it was to isolate that point (-1 = very anomalous, +1 = very normal)
  7. We convert this to a risk score 0-100 where 100 = highest risk

No labels needed. No train/test split in the traditional sense.
Validation is RETROSPECTIVE: we check if high-risk scores appear BEFORE known crashes.
"""

import numpy as np
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

# scikit-learn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ─── Configuration ────────────────────────────────────────────────────────────

# Features to use for the model (numeric only, no categorical)
# These are the features that showed signal in our analysis
NUMERIC_FEATURES = [
    "error_count_5m",
    "error_count_1h",
    "fatal_count_1h",
    "warn_count_1h",
    "distinct_errors_5m",
    "error_rate_delta",
    "has_stack_trace_5m",
    "state_transitions_1h",
    "restarts_24h",
    "pool_exhaustion_count_5m",
    "bus_error_count_5m",
]

# Features available but excluded (and why):
# - zkc_error_count_5m:        always zero for arcm_s (ZKC errors in separate file)
# - minutes_since_last_restart: highly variable scale, mostly noise (5000+ minutes)
# - heap_committed_mb:          only available for Nov 2025 (partial coverage)
# - cache_avg_get_time_ms:      all zeros (CSV is from abs_s, not arcm_s)
# - login_*:                    all zeros (same reason)
# - http_pool_*:                all zeros (same reason)
# - current_state:              categorical — encoded separately below

# State encoding: one-hot would add 6 sparse columns.
# Instead we use a single numeric "state_risk" feature:
STATE_RISK = {
    "STARTED": 0,     # normal
    "STOPPED": 1,     # stopped but controlled
    "STOPPING": 2,    # shutting down
    "STARTING": 3,    # booting — vulnerable period
    "DOWN": 4,        # crashed
    "FAILED": 5,      # repeated crash — worst
    "UNKNOWN": 1,     # conservative
    "DEACTIVATED": 0, # intentional
    "RECONFIGURING": 2,
    "UPDATING": 2,
}


# ─── Risk score output ────────────────────────────────────────────────────────

@dataclass
class RiskScore:
    """Risk assessment for one runnable at one point in time."""
    runnable: str
    predicted_at: datetime
    predicted_at_epoch: int
    risk_score: float          # 0.0 to 100.0
    risk_level: str            # GREEN, YELLOW, RED
    anomaly_score_raw: float   # raw Isolation Forest score (-1 to 1)
    top_features: list = field(default_factory=list)  # [{feature, value, contribution}]
    model_version: str = ""

    def to_dict(self) -> dict:
        return {
            "runnable": self.runnable,
            "predicted_at": self.predicted_at.isoformat(),
            "predicted_at_epoch": self.predicted_at_epoch,
            "risk_score": round(self.risk_score, 2),
            "risk_level": self.risk_level,
            "anomaly_score_raw": round(self.anomaly_score_raw, 4),
            "top_features": self.top_features,
            "model_version": self.model_version,
        }


# ─── Predictor ────────────────────────────────────────────────────────────────

class Predictor:
    """
    The B6 Predictor. Trains an Isolation Forest on feature vectors
    and produces risk scores.

    Usage:
        predictor = Predictor()
        predictor.fit(feature_vectors)           # list of FeatureVector
        risk_scores = predictor.predict(feature_vectors)  # list of RiskScore
        predictor.save("model_v1")               # save model to disk
        predictor.load("model_v1")               # load model from disk
    """

    def __init__(
        self,
        contamination: float = 0.02,   # expected fraction of anomalies (~2%)
        n_estimators: int = 200,        # number of trees
        random_state: int = 42,
        red_threshold: float = 70.0,    # risk_score >= 70 → RED
        yellow_threshold: float = 30.0, # risk_score >= 30 → YELLOW
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.red_threshold = red_threshold
        self.yellow_threshold = yellow_threshold

        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names = NUMERIC_FEATURES + ["state_risk"]
        self.model_version = ""
        self._is_fitted = False

    def _extract_matrix(self, feature_vectors) -> np.ndarray:
        """Convert FeatureVector list to numpy matrix."""
        rows = []
        for fv in feature_vectors:
            row = []
            for feat in NUMERIC_FEATURES:
                val = getattr(fv, feat, 0)
                # Replace -1 defaults with 0
                if val == -1 or val == -1.0:
                    val = 0
                row.append(float(val))
            # Add state_risk encoding
            state = getattr(fv, "current_state", "UNKNOWN")
            row.append(float(STATE_RISK.get(state, 1)))
            rows.append(row)
        return np.array(rows)

    def fit(self, feature_vectors, model_version: str = "v1"):
        """
        Train the Isolation Forest on feature vectors.

        Args:
            feature_vectors: list of FeatureVector objects
            model_version: version string for tracking
        """
        X = self._extract_matrix(feature_vectors)

        # Scale features (important for Isolation Forest to treat all features equally)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,  # use all CPU cores
        )
        self.model.fit(X_scaled)
        self.model_version = model_version
        self._is_fitted = True

        return self

    def predict(self, feature_vectors) -> list:
        """
        Predict risk scores for feature vectors.

        Returns:
            list of RiskScore objects
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self._extract_matrix(feature_vectors)
        X_scaled = self.scaler.transform(X)

        # Get raw anomaly scores
        # score_samples returns negative values: more negative = more anomalous
        raw_scores = self.model.score_samples(X_scaled)

        # Convert to 0-100 risk score
        # score_samples range is roughly -0.5 (very anomalous) to 0.5 (very normal)
        # We invert and rescale
        risk_scores = self._normalize_scores(raw_scores)

        results = []
        for i, fv in enumerate(feature_vectors):
            risk = risk_scores[i]
            level = "RED" if risk >= self.red_threshold else \
                    "YELLOW" if risk >= self.yellow_threshold else "GREEN"

            # Compute feature contributions (which features drove this score)
            top_feats = self._explain(X_scaled[i], X[i])

            results.append(RiskScore(
                runnable=fv.runnable,
                predicted_at=fv.window_end,
                predicted_at_epoch=fv.window_end_epoch,
                risk_score=risk,
                risk_level=level,
                anomaly_score_raw=float(raw_scores[i]),
                top_features=top_feats,
                model_version=self.model_version,
            ))

        return results

    def _normalize_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Convert raw Isolation Forest scores to 0-100 risk scale.

        Strategy: use percentile-based normalization.
        The most anomalous points (lowest raw scores) get risk ~100.
        The most normal points (highest raw scores) get risk ~0.
        """
        # Invert: lower raw = higher risk
        inverted = -raw_scores

        # Percentile-based scaling
        p_min = np.percentile(inverted, 1)   # 1st percentile (most normal)
        p_max = np.percentile(inverted, 99)  # 99th percentile (most anomalous)

        if p_max - p_min < 1e-10:
            return np.zeros_like(raw_scores)

        scaled = (inverted - p_min) / (p_max - p_min) * 100
        return np.clip(scaled, 0, 100)

    def _explain(self, x_scaled: np.ndarray, x_original: np.ndarray) -> list:
        """
        Simple feature importance explanation.
        Returns the top features that deviate most from the mean (in scaled space).
        """
        # How far each feature is from the training mean (in standard deviations)
        deviations = np.abs(x_scaled)

        # Get top 5 features by deviation
        top_indices = np.argsort(deviations)[::-1][:5]

        explanations = []
        for idx in top_indices:
            if deviations[idx] > 0.5:  # only include if meaningfully deviated
                explanations.append({
                    "feature": self.feature_names[idx],
                    "value": float(x_original[idx]),
                    "deviation_std": round(float(deviations[idx]), 2),
                })

        return explanations

    def get_stats(self, risk_scores: list) -> dict:
        """Compute statistics on risk scores."""
        scores = [rs.risk_score for rs in risk_scores]
        levels = [rs.risk_level for rs in risk_scores]

        return {
            "total_predictions": len(risk_scores),
            "mean_risk": round(float(np.mean(scores)), 2),
            "max_risk": round(float(np.max(scores)), 2),
            "red_count": levels.count("RED"),
            "yellow_count": levels.count("YELLOW"),
            "green_count": levels.count("GREEN"),
            "red_pct": round(levels.count("RED") / len(levels) * 100, 2),
            "model_version": self.model_version,
        }

    def save(self, filepath: str):
        """Save model and scaler to disk."""
        import pickle
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_version": self.model_version,
            "contamination": self.contamination,
            "red_threshold": self.red_threshold,
            "yellow_threshold": self.yellow_threshold,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """Load model and scaler from disk."""
        import pickle
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.model_version = data["model_version"]
        self.contamination = data["contamination"]
        self.red_threshold = data["red_threshold"]
        self.yellow_threshold = data["yellow_threshold"]
        self._is_fitted = True
