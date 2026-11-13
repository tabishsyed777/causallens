#!/usr/bin/env python3
"""Estimate a binary treatment effect with diagnostics and a doubly robust estimator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _smd(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left) or not len(right):
        return 0.0
    pooled = np.sqrt((np.var(left, ddof=1) + np.var(right, ddof=1)) / 2) if len(left) > 1 and len(right) > 1 else 0.0
    return float((np.mean(left) - np.mean(right)) / pooled) if pooled > 1e-12 else 0.0


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = np.sum(weights)
    return float(np.sum(values * weights) / total) if total > 1e-12 else float(np.mean(values))


def _parse_features(frame: pd.DataFrame, treatment: str, outcome: str, selected: str | None) -> list[str]:
    names = [name.strip() for name in selected.split(",")] if selected else [name for name in frame.columns if name not in {treatment, outcome}]
    if not names:
        raise ValueError("at least one numeric feature is required")
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"feature columns not found: {', '.join(missing)}")
    for name in names:
        if not pd.api.types.is_numeric_dtype(frame[name]):
            raise ValueError(f"feature column must be numeric: {name}")
    return names


def analyze(frame: pd.DataFrame, treatment: str, outcome: str, selected: str | None, seed: int, bootstrap: int) -> dict[str, Any]:
    if treatment not in frame.columns or outcome not in frame.columns:
        raise ValueError(f"CSV must contain '{treatment}' and '{outcome}' columns")
    if not pd.api.types.is_numeric_dtype(frame[treatment]) or not pd.api.types.is_numeric_dtype(frame[outcome]):
        raise ValueError("treatment and outcome columns must be numeric")
    data = frame.dropna(subset=[treatment, outcome]).copy()
    treatments = data[treatment].astype(float).to_numpy()
    if not set(np.unique(treatments)).issubset({0.0, 1.0}) or len(np.unique(treatments)) < 2:
        raise ValueError("treatment must contain both 0 and 1")
    features = _parse_features(data, treatment, outcome, selected)
    x_frame = data[features].apply(pd.to_numeric, errors="coerce")
    missing_counts = x_frame.isna().sum().to_dict()
    x = x_frame.fillna(x_frame.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=float)
    y = data[outcome].astype(float).to_numpy()
    treated_mask = treatments == 1
    control_mask = ~treated_mask
    warnings: list[str] = []

    propensity_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    propensity_model.fit(x, treatments)
    propensity = np.clip(propensity_model.predict_proba(x)[:, 1], 0.02, 0.98)
    mu0_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x[control_mask], y[control_mask])
    mu1_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x[treated_mask], y[treated_mask])
    mu0, mu1 = mu0_model.predict(x), mu1_model.predict(x)
    pseudo = (mu1 - mu0) + treatments * (y - mu1) / propensity - (1 - treatments) * (y - mu0) / (1 - propensity)
    ate = float(np.mean(pseudo))
    if bootstrap < 50:
        raise ValueError("bootstrap count must be at least 50")
    rng = np.random.default_rng(seed)
    draws = np.array([np.mean(pseudo[rng.integers(0, len(pseudo), len(pseudo))]) for _ in range(bootstrap)])
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    outcome_std = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    balance = []
    for column_index, feature in enumerate(features):
        before = _smd(x[treated_mask, column_index], x[control_mask, column_index])
        after_treated = _weighted_mean(x[treated_mask, column_index], 1 / propensity[treated_mask])
        after_control = _weighted_mean(x[control_mask, column_index], 1 / (1 - propensity[control_mask]))
        scale = np.sqrt((np.var(x[treated_mask, column_index]) + np.var(x[control_mask, column_index])) / 2)
        after = float((after_treated - after_control) / scale) if scale > 1e-12 else 0.0
        balance.append({"feature": feature, "smd_before": round(before, 4), "smd_after": round(after, 4)})
        if abs(before) > 0.1:
            warnings.append(f"{feature} has standardized mean difference {before:.2f} before weighting")
    if len(y[treated_mask]) < 20 or len(y[control_mask]) < 20:
        warnings.append("one or both treatment groups have fewer than 20 rows; uncertainty may be understated")
    if np.min(propensity) <= 0.03 or np.max(propensity) >= 0.97:
        warnings.append("propensity scores approach the overlap boundary; interpret the estimate cautiously")
    if sum(missing_counts.values()) > 0:
        warnings.append("missing feature values were median-imputed for model fitting")
    sensitivity = [{"delta_outcome_sd": delta, "lower_scenario": round(ate - delta * outcome_std, 4), "upper_scenario": round(ate + delta * outcome_std, 4)} for delta in (0.25, 0.5, 1.0)]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "doubly_robust_pseudo_outcome",
        "design": {"treatment": treatment, "outcome": outcome, "features": features, "bootstrap_draws": bootstrap},
        "sample": {"rows_used": int(len(y)), "treated": int(np.sum(treated_mask)), "control": int(np.sum(control_mask)), "outcome_mean": round(float(np.mean(y)), 4), "outcome_std": round(outcome_std, 4)},
        "estimate": {"ate": round(ate, 4), "bootstrap_ci_95": [round(float(ci_low), 4), round(float(ci_high), 4)], "propensity_min": round(float(np.min(propensity)), 4), "propensity_max": round(float(np.max(propensity)), 4)},
        "balance": balance,
        "sensitivity_stress": sensitivity,
        "warnings": sorted(set(warnings)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV with treatment, outcome, and numeric feature columns")
    parser.add_argument("--treatment", default="treatment")
    parser.add_argument("--outcome", default="outcome")
    parser.add_argument("--features", help="comma-separated feature column names")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")
    try:
        report = analyze(pd.read_csv(args.input), args.treatment, args.outcome, args.features, args.seed, args.bootstrap)
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
