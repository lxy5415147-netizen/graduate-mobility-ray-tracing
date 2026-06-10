# -*- coding: utf-8 -*-
"""
Direct refraction/scattering kernel for A-type graduate mobility beams.

This experiment is intentionally separated from the main O-U medium response
model. The main model estimates how much incident flow becomes A/B/C/D/E. This
script only studies where the A-type refraction flow goes:

    P(D | O, U, refraction)

The fitted score is normalized over legal destination cities for each O-U beam,
so the predicted A-flow distribution is conserved within each beam.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


O_CODE_RAW = "A_Daima_New"
U_CODE_RAW = "B_Daima_New"
D_CODE_RAW = "C_Daima_New"
FLOW_RAW = "TotalFlows"

O_CITY_RAW = "老家所在城市名称"
U_CITY_RAW = "高校所在城市"
D_CITY_RAW = "去向所在城市名称"
O_LEVEL_RAW = "老家所在城市等级"
U_LEVEL_RAW = "高校城市等级"
D_LEVEL_RAW = "去向所在城市等级"

RAW_COLS = [
    O_CODE_RAW,
    U_CODE_RAW,
    D_CODE_RAW,
    FLOW_RAW,
    O_CITY_RAW,
    U_CITY_RAW,
    D_CITY_RAW,
    O_LEVEL_RAW,
    U_LEVEL_RAW,
    D_LEVEL_RAW,
    "A_Province1",
    "B_Province1",
    "C_Province1",
    "A_Level2",
    "B_Level2",
    "C_Level2",
    "C_popu",
    "C_hos",
    "C_tea",
    "C_gdp",
    "C_aqi",
    "C_house",
    "C_muse",
    "C_cen",
    "C_Green_Rate",
    "C_Tertiary_rate",
    "C_avg_Num_patent",
    "C_avg_num_Lib",
]

LEVEL_SCORE = {
    "一线城市": 6,
    "新一线城市": 5,
    "二线城市": 4,
    "三线城市": 3,
    "四线城市": 2,
    "五线城市": 1,
}

FEATURE_COLS = [
    "log_D_popu",
    "log_D_gdp",
    "log_D_house",
    "D_tertiary",
    "D_center",
    "D_patent",
    "D_library",
    "D_hospital",
    "D_teacher",
    "D_museum",
    "D_green_rate",
    "D_aqi",
    "D_level",
    "D_minus_U_level",
    "D_minus_O_level",
    "abs_D_minus_U_level",
    "abs_D_minus_O_level",
    "log_UD_dist",
    "log_OD_dist",
    "log_total_path_dist",
    "same_OD_province",
    "same_UD_province",
    "same_OD_geo",
    "same_UD_geo",
    "target_cross_section_seed",
    "D_train_prior_log_flow",
]

EPS = 1e-12
EARTH_RADIUS_KM = 6371.0088


@dataclass
class PreparedData:
    a_oud: pd.DataFrame
    ou_beams: pd.DataFrame
    candidates: pd.DataFrame
    city_xy: pd.DataFrame


def code6(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".")[0]
    digits = re.sub(r"\D", "", text)
    if not digits:
        return ""
    return digits.zfill(6)[:6]


def read_csv_safely(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, usecols=usecols, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"Cannot read CSV: {path}")


def haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    lon1r = np.radians(lon1.astype(float))
    lat1r = np.radians(lat1.astype(float))
    lon2r = np.radians(lon2.astype(float))
    lat2r = np.radians(lat2.astype(float))
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def direction_change_from_sides(ou: np.ndarray, ud: np.ndarray, od: np.ndarray) -> np.ndarray:
    denom = np.clip(2.0 * ou * ud, EPS, None)
    cos_interior = np.clip((ou**2 + ud**2 - od**2) / denom, -1.0, 1.0)
    interior = np.arccos(cos_interior)
    # 0 means the outgoing ray continues the incoming O->U direction.
    return np.pi - interior


def load_city_coordinates(coord_path: Path) -> pd.DataFrame:
    coords = read_csv_safely(coord_path)
    if "Prefectu_1" in coords.columns:
        code_col = "Prefectu_1"
    elif "Prefectu_2" in coords.columns:
        code_col = "Prefectu_2"
    else:
        code_col = "PrefectureCode"
    name_col = "Prefecture" if "Prefecture" in coords.columns else coords.columns[1]
    out = coords[[code_col, name_col, "Longitude", "Latitude"]].copy()
    out.columns = ["city_code", "coord_city", "lon", "lat"]
    out["city_code"] = out["city_code"].map(code6)
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out = out[(out["city_code"] != "") & out["lon"].notna() & out["lat"].notna()].drop_duplicates("city_code")
    aliases = {
        "110000": "110100",
        "120000": "120100",
        "310000": "310100",
        "500000": "500100",
        "540600": "542400",
    }
    alias_rows = []
    existing = set(out["city_code"])
    for alias, source in aliases.items():
        source_row = out[out["city_code"] == source]
        if not source_row.empty and alias not in existing:
            row = source_row.iloc[0].copy()
            row["city_code"] = alias
            alias_rows.append(row)
    if alias_rows:
        out = pd.concat([out, pd.DataFrame(alias_rows)], ignore_index=True)
    manual_rows = [
        {"city_code": "429005", "coord_city": "潜江市", "lon": 112.8993, "lat": 30.4015},
        {"city_code": "469006", "coord_city": "万宁市", "lon": 110.3897, "lat": 18.7953},
    ]
    missing_manual = [row for row in manual_rows if row["city_code"] not in set(out["city_code"])]
    if missing_manual:
        out = pd.concat([out, pd.DataFrame(missing_manual)], ignore_index=True)
    return out


def prepare_data(input_path: Path, coord_path: Path, min_ou_flow: float) -> PreparedData:
    all_cols = list(pd.read_csv(input_path, nrows=0, encoding="utf-8-sig").columns)
    usecols = [c for c in RAW_COLS if c in all_cols]
    raw = read_csv_safely(input_path, usecols=usecols)
    raw["O_code"] = raw[O_CODE_RAW].map(code6)
    raw["U_code"] = raw[U_CODE_RAW].map(code6)
    raw["D_code"] = raw[D_CODE_RAW].map(code6)
    raw["flow"] = pd.to_numeric(raw[FLOW_RAW], errors="coerce").fillna(0.0)
    raw = raw[(raw["flow"] > 0) & (raw["O_code"] != "") & (raw["U_code"] != "") & (raw["D_code"] != "")].copy()

    for raw_col, new_col in [
        (O_CITY_RAW, "O_city"),
        (U_CITY_RAW, "U_city"),
        (D_CITY_RAW, "D_city"),
        (O_LEVEL_RAW, "O_level_text"),
        (U_LEVEL_RAW, "U_level_text"),
        (D_LEVEL_RAW, "D_level_text"),
    ]:
        raw[new_col] = raw[raw_col].astype(str).str.strip() if raw_col in raw.columns else ""

    raw["O_level"] = raw["O_level_text"].map(LEVEL_SCORE).astype(float)
    raw["U_level"] = raw["U_level_text"].map(LEVEL_SCORE).astype(float)
    raw["D_level"] = raw["D_level_text"].map(LEVEL_SCORE).astype(float)

    a_rows = raw[(raw["O_code"] != raw["U_code"]) & (raw["D_code"] != raw["O_code"]) & (raw["D_code"] != raw["U_code"])].copy()

    group_cols = [
        "O_code",
        "O_city",
        "U_code",
        "U_city",
        "D_code",
        "D_city",
        "O_level",
        "U_level",
        "D_level",
        "A_Province1",
        "B_Province1",
        "C_Province1",
        "A_Level2",
        "B_Level2",
        "C_Level2",
    ]
    d_attr_cols = [
        "C_popu",
        "C_hos",
        "C_tea",
        "C_gdp",
        "C_aqi",
        "C_house",
        "C_muse",
        "C_cen",
        "C_Green_Rate",
        "C_Tertiary_rate",
        "C_avg_Num_patent",
        "C_avg_num_Lib",
    ]
    present_group_cols = [c for c in group_cols if c in a_rows.columns]
    present_attr_cols = [c for c in d_attr_cols if c in a_rows.columns]
    agg_spec = {"flow": "sum"}
    agg_spec.update({c: "first" for c in present_attr_cols})
    a_oud = a_rows.groupby(present_group_cols, as_index=False, dropna=False).agg(agg_spec)

    ou_cols = [
        "O_code",
        "O_city",
        "U_code",
        "U_city",
        "O_level",
        "U_level",
        "A_Province1",
        "B_Province1",
        "A_Level2",
        "B_Level2",
    ]
    present_ou_cols = [c for c in ou_cols if c in a_oud.columns]
    ou_beams = a_oud.groupby(present_ou_cols, as_index=False, dropna=False)["flow"].sum().rename(columns={"flow": "F_A"})
    ou_beams = ou_beams[ou_beams["F_A"] >= min_ou_flow].copy()
    a_oud = a_oud.merge(ou_beams[["O_code", "U_code", "F_A"]], on=["O_code", "U_code"], how="inner")
    a_oud["obs_rate"] = a_oud["flow"] / a_oud["F_A"].clip(lower=EPS)

    cand_cols = ["D_code", "D_city", "D_level", "C_Province1", "C_Level2"] + present_attr_cols
    candidates = (
        a_oud.sort_values("flow", ascending=False)[cand_cols]
        .drop_duplicates("D_code")
        .reset_index(drop=True)
    )

    city_xy = load_city_coordinates(coord_path)
    needed_codes = pd.concat([a_oud["O_code"], a_oud["U_code"], candidates["D_code"]]).drop_duplicates()
    city_xy = city_xy[city_xy["city_code"].isin(needed_codes)].copy()

    for label, frame, col in [("O", a_oud, "O_code"), ("U", a_oud, "U_code"), ("D", candidates, "D_code")]:
        missing = sorted(set(frame[col]) - set(city_xy["city_code"]))
        if missing:
            print(f"[warn] {label} coordinate missing: {len(missing)} cities; examples={missing[:8]}")

    return PreparedData(a_oud=a_oud, ou_beams=ou_beams, candidates=candidates, city_xy=city_xy)


def add_destination_prior(candidates: pd.DataFrame, train_pos: pd.DataFrame) -> pd.DataFrame:
    """Add a train-only destination cross-section prior to candidate cities."""
    out = candidates.copy()
    prior = train_pos.groupby("D_code", as_index=False)["flow"].sum().rename(columns={"flow": "D_train_flow"})
    out = out.merge(prior, on="D_code", how="left")
    fallback = float(max(train_pos["flow"].median(), 1.0)) if len(train_pos) else 1.0
    out["D_train_flow"] = out["D_train_flow"].fillna(fallback)
    out["D_train_prior_log_flow"] = np.log1p(out["D_train_flow"].clip(lower=0))
    return out.drop(columns=["D_train_flow"])


def attach_coordinates(df: pd.DataFrame, city_xy: pd.DataFrame, code_col: str, prefix: str) -> pd.DataFrame:
    coords = city_xy.rename(
        columns={
            "city_code": code_col,
            "coord_city": f"{prefix}_coord_city",
            "lon": f"{prefix}_lon",
            "lat": f"{prefix}_lat",
        }
    )
    return df.merge(coords[[code_col, f"{prefix}_lon", f"{prefix}_lat"]], on=code_col, how="left")


def build_feature_frame(rows: pd.DataFrame, candidates: pd.DataFrame, city_xy: pd.DataFrame) -> pd.DataFrame:
    candidate_payload = [c for c in candidates.columns if c != "D_code"]
    rows = rows.drop(columns=[c for c in candidate_payload if c in rows.columns], errors="ignore")
    rows = rows.merge(candidates, on="D_code", how="left")

    out = rows.copy()
    out = attach_coordinates(out, city_xy, "O_code", "O")
    out = attach_coordinates(out, city_xy, "U_code", "U")
    out = attach_coordinates(out, city_xy, "D_code", "D")
    out = out.dropna(subset=["O_lon", "O_lat", "U_lon", "U_lat", "D_lon", "D_lat"]).copy()

    od = haversine_km(out["O_lon"].to_numpy(), out["O_lat"].to_numpy(), out["D_lon"].to_numpy(), out["D_lat"].to_numpy())
    ud = haversine_km(out["U_lon"].to_numpy(), out["U_lat"].to_numpy(), out["D_lon"].to_numpy(), out["D_lat"].to_numpy())
    ou = haversine_km(out["O_lon"].to_numpy(), out["O_lat"].to_numpy(), out["U_lon"].to_numpy(), out["U_lat"].to_numpy())
    out["OD_dist"] = od
    out["UD_dist"] = ud
    out["OU_dist"] = ou
    out["log_UD_dist"] = np.log1p(ud)
    out["log_OD_dist"] = np.log1p(od)
    out["log_total_path_dist"] = np.log1p(ou + ud)

    numeric_map = {
        "log_D_popu": ("C_popu", True),
        "log_D_gdp": ("C_gdp", True),
        "log_D_house": ("C_house", True),
        "D_tertiary": ("C_Tertiary_rate", False),
        "D_center": ("C_cen", False),
        "D_patent": ("C_avg_Num_patent", False),
        "D_library": ("C_avg_num_Lib", False),
        "D_hospital": ("C_hos", False),
        "D_teacher": ("C_tea", False),
        "D_museum": ("C_muse", False),
        "D_green_rate": ("C_Green_Rate", False),
        "D_aqi": ("C_aqi", False),
    }
    for new_col, (old_col, log_transform) in numeric_map.items():
        s = pd.to_numeric(out.get(old_col), errors="coerce")
        if log_transform:
            s = np.log1p(s.clip(lower=0))
        out[new_col] = s

    out["D_level"] = pd.to_numeric(out["D_level"], errors="coerce")
    out["O_level"] = pd.to_numeric(out["O_level"], errors="coerce")
    out["U_level"] = pd.to_numeric(out["U_level"], errors="coerce")
    out["D_minus_U_level"] = out["D_level"] - out["U_level"]
    out["D_minus_O_level"] = out["D_level"] - out["O_level"]
    out["abs_D_minus_U_level"] = out["D_minus_U_level"].abs()
    out["abs_D_minus_O_level"] = out["D_minus_O_level"].abs()

    out["same_OD_province"] = (out["A_Province1"].astype(str) == out["C_Province1"].astype(str)).astype(float)
    out["same_UD_province"] = (out["B_Province1"].astype(str) == out["C_Province1"].astype(str)).astype(float)
    out["same_OD_geo"] = (out["A_Level2"].astype(str) == out["C_Level2"].astype(str)).astype(float)
    out["same_UD_geo"] = (out["B_Level2"].astype(str) == out["C_Level2"].astype(str)).astype(float)

    out["target_cross_section_seed"] = (
        out["log_D_popu"].fillna(0)
        + out["log_D_gdp"].fillna(0)
        + out["D_tertiary"].fillna(0)
        + out["D_center"].fillna(0)
        + out["D_patent"].fillna(0)
    )
    return out


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    features = frame[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features.to_numpy(dtype=float)


def iter_chunks(df: pd.DataFrame, chunk_size: int):
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start : start + chunk_size].copy()


def sample_negative_destinations(
    base: pd.DataFrame,
    candidate_codes: np.ndarray,
    neg_per_pos: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(base)
    sampled = rng.choice(candidate_codes, size=(n, neg_per_pos), replace=True)
    o = base["O_code"].to_numpy()[:, None]
    u = base["U_code"].to_numpy()[:, None]
    d = base["D_code"].to_numpy()[:, None]
    bad = (sampled == o) | (sampled == u) | (sampled == d)
    for _ in range(20):
        if not bad.any():
            break
        sampled[bad] = rng.choice(candidate_codes, size=int(bad.sum()), replace=True)
        bad = (sampled == o) | (sampled == u) | (sampled == d)

    neg = base.loc[base.index.repeat(neg_per_pos)].copy()
    neg["D_code"] = sampled.reshape(-1)
    neg = neg[(neg["D_code"] != neg["O_code"]) & (neg["D_code"] != neg["U_code"])].copy()
    return neg


def make_training_batch(
    pos: pd.DataFrame,
    candidates: pd.DataFrame,
    city_xy: pd.DataFrame,
    candidate_codes: np.ndarray,
    neg_per_pos: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_part = pos.copy()
    pos_part["label"] = 1
    pos_part["sample_weight"] = np.log1p(pos_part["flow"].astype(float)).clip(lower=1.0)

    neg_part = sample_negative_destinations(pos, candidate_codes, neg_per_pos, rng)
    neg_part["label"] = 0
    neg_part["sample_weight"] = np.log1p(neg_part["flow"].astype(float)).clip(lower=1.0) / max(neg_per_pos, 1)

    batch = pd.concat([pos_part, neg_part], ignore_index=True, sort=False)
    features = build_feature_frame(batch, candidates, city_xy)
    x = feature_matrix(features)
    y = features["label"].to_numpy(dtype=int)
    w = features["sample_weight"].to_numpy(dtype=float)
    return x, y, w


def fit_negative_sampled_kernel(
    train_pos: pd.DataFrame,
    candidates: pd.DataFrame,
    city_xy: pd.DataFrame,
    neg_per_pos: int,
    chunk_size: int,
    epochs: int,
    alpha: float,
    seed: int,
) -> tuple[SGDClassifier, StandardScaler]:
    rng = np.random.default_rng(seed)
    candidate_codes = candidates["D_code"].to_numpy()
    scaler = StandardScaler()

    for chunk in iter_chunks(train_pos, chunk_size):
        x, _, _ = make_training_batch(chunk, candidates, city_xy, candidate_codes, neg_per_pos, rng)
        if len(x):
            scaler.partial_fit(x)

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=1,
        tol=None,
        learning_rate="optimal",
        random_state=seed,
        fit_intercept=True,
    )

    first = True
    for epoch in range(epochs):
        shuffled = train_pos.sample(frac=1.0, random_state=seed + epoch).reset_index(drop=True)
        for chunk in iter_chunks(shuffled, chunk_size):
            x, y, w = make_training_batch(chunk, candidates, city_xy, candidate_codes, neg_per_pos, rng)
            if not len(x):
                continue
            xz = scaler.transform(x)
            if first:
                clf.partial_fit(xz, y, classes=np.array([0, 1]), sample_weight=w)
                first = False
            else:
                clf.partial_fit(xz, y, sample_weight=w)
    return clf, scaler


def make_candidate_panel(ou_chunk: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    left = ou_chunk.loc[ou_chunk.index.repeat(len(candidates))].reset_index(drop=True)
    right = pd.concat([candidates] * len(ou_chunk), ignore_index=True)
    panel = pd.concat([left.reset_index(drop=True), right.reset_index(drop=True)], axis=1)
    panel = panel[(panel["D_code"] != panel["O_code"]) & (panel["D_code"] != panel["U_code"])].copy()
    return panel


def softmax_by_beam(panel: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    out = panel[["O_code", "U_code", "D_code", "D_city", "D_level", "UD_dist", "OD_dist"]].copy()
    out["score"] = scores
    keys = ["O_code", "U_code"]
    max_score = out.groupby(keys)["score"].transform("max")
    out["exp_score"] = np.exp(out["score"] - max_score)
    denom = out.groupby(keys)["exp_score"].transform("sum").clip(lower=EPS)
    out["pred_prob"] = out["exp_score"] / denom
    return out.drop(columns=["exp_score"])


def evaluate_predictions(
    clf: SGDClassifier,
    scaler: StandardScaler,
    ou_eval: pd.DataFrame,
    observed: pd.DataFrame,
    candidates: pd.DataFrame,
    city_xy: pd.DataFrame,
    output_dir: Path,
    label: str,
    panel_chunk_ou: int,
    top_k: int,
) -> dict[str, float]:
    obs_cols = ["O_code", "U_code", "D_code", "flow", "obs_rate"]
    obs = observed[obs_cols].copy()
    dominant = (
        obs.sort_values("flow", ascending=False)
        .drop_duplicates(["O_code", "U_code"])
        [["O_code", "U_code", "D_code"]]
        .rename(columns={"D_code": "dominant_D_obs"})
    )

    per_ou_rows: list[pd.DataFrame] = []
    top_rows: list[pd.DataFrame] = []
    total_flow = 0.0
    total_nll = 0.0
    total_tv = 0.0
    total_obs_ud = 0.0
    total_pred_ud = 0.0
    total_obs_level = 0.0
    total_pred_level = 0.0

    for ou_chunk in iter_chunks(ou_eval.reset_index(drop=True), panel_chunk_ou):
        panel = make_candidate_panel(ou_chunk, candidates)
        feats = build_feature_frame(panel, candidates, city_xy)
        if feats.empty:
            continue
        x = scaler.transform(feature_matrix(feats))
        scores = clf.decision_function(x)
        pred = softmax_by_beam(feats, scores)
        pred = pred.merge(obs, on=["O_code", "U_code", "D_code"], how="left")
        pred["flow"] = pred["flow"].fillna(0.0)
        pred["obs_rate"] = pred["obs_rate"].fillna(0.0)

        pred["abs_err"] = (pred["pred_prob"] - pred["obs_rate"]).abs()
        pred["nll_part"] = np.where(pred["obs_rate"] > 0, -pred["flow"] * np.log(pred["pred_prob"].clip(lower=EPS)), 0.0)
        pred["obs_ud_part"] = pred["flow"] * pred["UD_dist"]
        pred["pred_ud_part"] = pred["pred_prob"] * pred["UD_dist"]
        pred["obs_level_part"] = pred["flow"] * pred["D_level"]
        pred["pred_level_part"] = pred["pred_prob"] * pred["D_level"]

        grouped = pred.groupby(["O_code", "U_code"], as_index=False).agg(
            F_A=("flow", "sum"),
            tv_distance=("abs_err", lambda s: 0.5 * float(s.sum())),
            nll=("nll_part", "sum"),
            obs_mean_ud_km=("obs_ud_part", "sum"),
            pred_mean_ud_km=("pred_ud_part", "sum"),
            obs_mean_D_level=("obs_level_part", "sum"),
            pred_mean_D_level=("pred_level_part", "sum"),
        )
        grouped["obs_mean_ud_km"] = grouped["obs_mean_ud_km"] / grouped["F_A"].clip(lower=EPS)
        grouped["obs_mean_D_level"] = grouped["obs_mean_D_level"] / grouped["F_A"].clip(lower=EPS)
        grouped = grouped.merge(dominant, on=["O_code", "U_code"], how="left")

        top = (
            pred.sort_values(["O_code", "U_code", "pred_prob"], ascending=[True, True, False])
            .groupby(["O_code", "U_code"], as_index=False)
            .head(top_k)
            .copy()
        )
        top["rank"] = top.groupby(["O_code", "U_code"]).cumcount() + 1
        top = top.merge(dominant, on=["O_code", "U_code"], how="left")
        top["is_observed_dominant"] = (top["D_code"] == top["dominant_D_obs"]).astype(int)

        top1 = top[top["rank"] == 1][["O_code", "U_code", "D_code"]].rename(columns={"D_code": "top1_D_pred"})
        topk_hit = top.groupby(["O_code", "U_code"], as_index=False)["is_observed_dominant"].max().rename(
            columns={"is_observed_dominant": f"top{top_k}_hit"}
        )
        grouped = grouped.merge(top1, on=["O_code", "U_code"], how="left")
        grouped = grouped.merge(topk_hit, on=["O_code", "U_code"], how="left")
        grouped["top1_hit"] = (grouped["top1_D_pred"] == grouped["dominant_D_obs"]).astype(int)

        f = grouped["F_A"].sum()
        total_flow += float(f)
        total_nll += float(grouped["nll"].sum())
        total_tv += float((grouped["tv_distance"] * grouped["F_A"]).sum())
        total_obs_ud += float((grouped["obs_mean_ud_km"] * grouped["F_A"]).sum())
        total_pred_ud += float((grouped["pred_mean_ud_km"] * grouped["F_A"]).sum())
        total_obs_level += float((grouped["obs_mean_D_level"] * grouped["F_A"]).sum())
        total_pred_level += float((grouped["pred_mean_D_level"] * grouped["F_A"]).sum())

        per_ou_rows.append(grouped)
        top_rows.append(top)

    per_ou = pd.concat(per_ou_rows, ignore_index=True) if per_ou_rows else pd.DataFrame()
    top_pred = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()

    if not per_ou.empty:
        per_ou.to_csv(output_dir / f"{label}_per_ou_metrics.csv", index=False, encoding="utf-8-sig")
    if not top_pred.empty:
        top_pred.to_csv(output_dir / f"{label}_top{top_k}_destinations.csv", index=False, encoding="utf-8-sig")

    metrics = {
        f"{label}_ou_beams": float(len(per_ou)),
        f"{label}_A_flow": float(total_flow),
        f"{label}_weighted_nll": float(total_nll / max(total_flow, EPS)),
        f"{label}_weighted_tv_distance": float(total_tv / max(total_flow, EPS)),
        f"{label}_top1_dominant_accuracy": float(np.average(per_ou["top1_hit"], weights=per_ou["F_A"])) if not per_ou.empty else float("nan"),
        f"{label}_top{top_k}_dominant_coverage": float(np.average(per_ou[f"top{top_k}_hit"], weights=per_ou["F_A"])) if not per_ou.empty else float("nan"),
        f"{label}_obs_mean_ud_km": float(total_obs_ud / max(total_flow, EPS)),
        f"{label}_pred_mean_ud_km": float(total_pred_ud / max(total_flow, EPS)),
        f"{label}_obs_mean_D_level": float(total_obs_level / max(total_flow, EPS)),
        f"{label}_pred_mean_D_level": float(total_pred_level / max(total_flow, EPS)),
    }
    return metrics


def save_coefficients(clf: SGDClassifier, scaler: StandardScaler, output_dir: Path) -> None:
    coefs = clf.coef_.reshape(-1)
    rows = []
    for col, coef, mean, scale in zip(FEATURE_COLS, coefs, scaler.mean_, scaler.scale_):
        rows.append({"feature": col, "coef_standardized": coef, "scaler_mean": mean, "scaler_scale": scale})
    pd.DataFrame(rows).sort_values("coef_standardized", ascending=False).to_csv(
        output_dir / "model_coefficients.csv", index=False, encoding="utf-8-sig"
    )


def write_readme(output_dir: Path, metrics: dict[str, float], args: argparse.Namespace) -> None:
    lines = [
        "# Direct Refraction Kernel Experiment",
        "",
        "This folder contains the outputs of `run_direct_refraction_kernel.py`.",
        "",
        "Model target:",
        "",
        "```text",
        "P(D | O, U, refraction)",
        "```",
        "",
        "Only A-type beams are used: `O != U`, `D != O`, and `D != U`.",
        "",
        "The experiment uses negative-sampled destination scoring for training and",
        "full legal-candidate softmax normalization for evaluation.",
        "",
        "Key settings:",
        "",
        f"- negative candidates per positive: `{args.negatives}`",
        f"- epochs: `{args.epochs}`",
        f"- minimum A-flow per O-U beam: `{args.min_ou_flow}`",
        f"- random seed: `{args.seed}`",
        "",
        "Metrics:",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"- `{key}`: {value:.6f}")
    (output_dir / "README_outputs.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parents[1]
    return argparse.ArgumentParser(description="Run direct A-type refraction destination kernel.").parse_args()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct A-type refraction destination kernel.")
    base = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--input",
        type=Path,
        default=base / "talent_ray_2d_calibration" / "data" / "raw" / "ReadytoRunModel_OUD_CityLevel.csv",
    )
    parser.add_argument(
        "--coords",
        type=Path,
        default=base / "talent_ray_2d_calibration" / "data" / "raw" / "Prefecture_2017_coordination.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=25000)
    parser.add_argument("--panel-chunk-ou", type=int, default=250)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-ou-flow", type=float, default=1.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_data(args.input, args.coords, args.min_ou_flow)

    keys = prepared.ou_beams[["O_code", "U_code"]].copy()
    train_keys, test_keys = train_test_split(keys, test_size=args.test_size, random_state=args.seed)
    train_keys = train_keys.assign(split="train")
    test_keys = test_keys.assign(split="test")
    splits = pd.concat([train_keys, test_keys], ignore_index=True)

    a_oud = prepared.a_oud.merge(splits, on=["O_code", "U_code"], how="inner")
    train_pos = a_oud[a_oud["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    test_pos = a_oud[a_oud["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
    train_ou = prepared.ou_beams.merge(train_keys[["O_code", "U_code"]], on=["O_code", "U_code"], how="inner")
    test_ou = prepared.ou_beams.merge(test_keys[["O_code", "U_code"]], on=["O_code", "U_code"], how="inner")
    candidates = add_destination_prior(prepared.candidates, train_pos)

    print(f"A O-U-D rows: {len(prepared.a_oud):,}")
    print(f"A O-U beams: {len(prepared.ou_beams):,}")
    print(f"Candidate D cities: {len(prepared.candidates):,}")
    print(f"Train positive O-U-D rows: {len(train_pos):,}; test positive O-U-D rows: {len(test_pos):,}")

    clf, scaler = fit_negative_sampled_kernel(
        train_pos=train_pos,
        candidates=candidates,
        city_xy=prepared.city_xy,
        neg_per_pos=args.negatives,
        chunk_size=args.chunk_size,
        epochs=args.epochs,
        alpha=args.alpha,
        seed=args.seed,
    )

    metrics: dict[str, float] = {
        "a_oud_rows": float(len(prepared.a_oud)),
        "a_ou_beams": float(len(prepared.ou_beams)),
        "candidate_d_cities": float(len(prepared.candidates)),
        "train_positive_oud_rows": float(len(train_pos)),
        "test_positive_oud_rows": float(len(test_pos)),
    }
    metrics.update(
        evaluate_predictions(
            clf,
            scaler,
            test_ou,
            test_pos,
            candidates,
            prepared.city_xy,
            args.output_dir,
            "test",
            args.panel_chunk_ou,
            args.top_k,
        )
    )
    metrics.update(
        evaluate_predictions(
            clf,
            scaler,
            prepared.ou_beams,
            prepared.a_oud,
            candidates,
            prepared.city_xy,
            args.output_dir,
            "all",
            args.panel_chunk_ou,
            args.top_k,
        )
    )

    save_coefficients(clf, scaler, args.output_dir)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(args.output_dir, metrics, args)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
