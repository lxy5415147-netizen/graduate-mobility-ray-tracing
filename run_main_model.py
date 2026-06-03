# -*- coding: utf-8 -*-
"""
Main model: O-U conditional probabilistic medium response.

Core idea:
Because an aggregated talent beam contains individual heterogeneity, it does
not correspond to one deterministic outgoing path at the university-city
interface. Instead, it appears as a probability distribution over reflection,
absorption, and refraction response states.

This script implements the first-version main model only:
- ABC foreign incident response, O != U.
- DE local escape/retention response, O == U.

It does not implement baseline models, Ucity-only comparison, talent spectrum,
destination scattering, background light, or emission lobes.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split


O_CODE = "A_Daima_New"
U_CODE = "B_Daima_New"
D_CODE = "C_Daima_New"
FLOW = "TotalFlows"

O_CITY = "老家所在城市名称"
U_CITY = "高校所在城市"
O_LEVEL_TEXT = "老家所在城市等级"
U_LEVEL_TEXT = "高校城市等级"

ABC_CLASSES = ["refraction", "reflection", "absorption"]
DE_CLASSES = ["local_escape", "local_retention"]

TYPE_MAP = {
    "A_refraction": "refraction",
    "B_reflection": "reflection",
    "C_absorption": "absorption",
    "D_local_escape": "local_escape",
    "E_local_retention": "local_retention",
}

LEVEL_SCORE = {
    "一线城市": 6,
    "新一线城市": 5,
    "二线城市": 4,
    "三线城市": 3,
    "四线城市": 2,
    "五线城市": 1,
}

NUMERIC_PAIRS = [
    ("popu", "A_popu", "B_popu"),
    ("gdp", "A_gdp", "B_gdp"),
    ("house", "A_house", "B_house"),
    ("aqi", "A_aqi", "B_aqi"),
    ("hos", "A_hos", "B_hos"),
    ("tea", "A_tea", "B_tea"),
    ("muse", "A_muse", "B_muse"),
    ("cen", "A_cen", "B_cen"),
    ("Green_Rate", "A_Green_Rate", "B_Green_Rate"),
    ("Tertiary_rate", "A_Tertiary_rate", "B_Tertiary_rate"),
    ("Num_patent", "A_avg_Num_patent", "B_avg_Num_patent"),
    ("num_Lib", "A_avg_num_Lib", "B_avg_num_Lib"),
]

OPTIONAL_COLS = [
    "AB_Dist",
    "A_Province1",
    "B_Province1",
    "A_Level2",
    "B_Level2",
    "Same_Geo",
    "Same_Geo1",
    "Same_Prov",
    "高校档次",
    "档次",
]

EPS = 1e-12


@dataclass
class StructuredSoftmaxModel:
    class_names: list[str]
    feature_sets: dict[str, list[str]]
    means: dict[str, float]
    stds: dict[str, float]
    beta: np.ndarray
    slices: dict[str, slice]
    alpha_l2: float


def existing_columns(path: str | Path) -> list[str]:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return list(pd.read_csv(path, nrows=0, encoding=enc).columns)
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"Cannot inspect columns for {path}")


def needed_columns(all_cols: list[str]) -> list[str]:
    required = {O_CODE, U_CODE, D_CODE, FLOW, O_CITY, U_CITY, O_LEVEL_TEXT, U_LEVEL_TEXT}
    requested = set(required) | set(OPTIONAL_COLS)
    for _, a_col, b_col in NUMERIC_PAIRS:
        requested.add(a_col)
        requested.add(b_col)
    available = set(all_cols)
    missing = sorted(required - available)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    return [c for c in all_cols if c in requested]


def read_csv_safely(path: str | Path, usecols: list[str], sample_rows: int | None) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, usecols=usecols, nrows=sample_rows, encoding=enc, low_memory=False)
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"Failed to read {path}")


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


def classify_oud_type(df: pd.DataFrame) -> pd.Series:
    o, u, d = df["O_code"], df["U_code"], df["D_code"]
    labels = np.select(
        [
            (o != u) & (d != o) & (d != u),
            (o != u) & (d == o),
            (o != u) & (d == u),
            (o == u) & (d != u),
            (o == u) & (d == u),
        ],
        ["A_refraction", "B_reflection", "C_absorption", "D_local_escape", "E_local_retention"],
        default="Unknown",
    )
    return pd.Series(labels, index=df.index).map(lambda x: TYPE_MAP.get(x, "Unknown"))


def add_basic_fields(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out["O_code"] = out[O_CODE].map(code6)
    out["U_code"] = out[U_CODE].map(code6)
    out["D_code"] = out[D_CODE].map(code6)
    out["flow"] = pd.to_numeric(out[FLOW], errors="coerce").fillna(0.0)
    out = out[(out["flow"] > 0) & (out["O_code"] != "") & (out["U_code"] != "") & (out["D_code"] != "")].copy()
    out["response_type"] = classify_oud_type(out)
    out["O_city"] = out[O_CITY].astype(str).str.strip()
    out["U_city"] = out[U_CITY].astype(str).str.strip()
    out["O_level_score"] = out[O_LEVEL_TEXT].map(LEVEL_SCORE).astype(float)
    out["U_level_score"] = out[U_LEVEL_TEXT].map(LEVEL_SCORE).astype(float)
    out["level_gap_U_minus_O"] = out["U_level_score"] - out["O_level_score"]
    out["log_AB_Dist"] = np.log1p(pd.to_numeric(out.get("AB_Dist"), errors="coerce"))
    return out


def aggregate_abc(df: pd.DataFrame, min_total: float) -> pd.DataFrame:
    ext = df[(df["O_code"] != df["U_code"]) & (df["response_type"].isin(ABC_CLASSES))].copy()
    group_cols = ["O_code", "O_city", "U_code", "U_city"]
    feature_cols = (
        ["log_AB_Dist", "O_level_score", "U_level_score", "level_gap_U_minus_O", O_LEVEL_TEXT, U_LEVEL_TEXT]
        + [c for _, a_col, b_col in NUMERIC_PAIRS for c in (a_col, b_col)]
        + OPTIONAL_COLS
    )
    flows = ext.pivot_table(index=group_cols, columns="response_type", values="flow", aggfunc="sum", fill_value=0).reset_index()
    for label in ABC_CLASSES:
        if label not in flows.columns:
            flows[label] = 0.0
    flows["F_external"] = flows[ABC_CLASSES].sum(axis=1)
    features = ext.groupby(group_cols, as_index=False).agg({c: "first" for c in feature_cols if c in ext.columns})
    out = flows.merge(features, on=group_cols, how="left")
    out = out[out["F_external"] >= min_total].copy()
    out["log_input_flow"] = np.log1p(out["F_external"])
    for label in ABC_CLASSES:
        out[f"{label}_rate_obs"] = out[label] / out["F_external"].replace(0, np.nan)
    out["observed_dominant"] = out[ABC_CLASSES].idxmax(axis=1)
    return out


def aggregate_de(df: pd.DataFrame, min_total: float) -> pd.DataFrame:
    local = df[(df["O_code"] == df["U_code"]) & (df["response_type"].isin(DE_CLASSES))].copy()
    group_cols = ["U_code", "U_city"]
    feature_cols = ["U_level_score", U_LEVEL_TEXT] + [b_col for _, _, b_col in NUMERIC_PAIRS] + ["B_Province1", "B_Level2", "高校档次", "档次"]
    flows = local.pivot_table(index=group_cols, columns="response_type", values="flow", aggfunc="sum", fill_value=0).reset_index()
    for label in DE_CLASSES:
        if label not in flows.columns:
            flows[label] = 0.0
    flows["F_local"] = flows[DE_CLASSES].sum(axis=1)
    features = local.groupby(group_cols, as_index=False).agg({c: "first" for c in feature_cols if c in local.columns})
    out = flows.merge(features, on=group_cols, how="left")
    out = out[out["F_local"] >= min_total].copy()
    out["log_input_flow"] = np.log1p(out["F_local"])
    for label in DE_CLASSES:
        out[f"{label}_rate_obs"] = out[label] / out["F_local"].replace(0, np.nan)
    out["observed_dominant"] = out[DE_CLASSES].idxmax(axis=1)

    # Mirror U attributes into O fields only so the same feature builder can be
    # reused. These mirrored fields are not interpreted as a real O-U path.
    out["O_code"] = out["U_code"]
    out["log_AB_Dist"] = 0.0
    out["O_level_score"] = out["U_level_score"]
    out["level_gap_U_minus_O"] = 0.0
    out[O_LEVEL_TEXT] = out[U_LEVEL_TEXT]
    for _, a_col, b_col in NUMERIC_PAIRS:
        out[a_col] = out[b_col] if b_col in out.columns else np.nan
    out["A_Province1"] = out["B_Province1"] if "B_Province1" in out.columns else ""
    out["A_Level2"] = out["B_Level2"] if "B_Level2" in out.columns else ""
    out["Same_Prov"] = "Yes"
    out["Same_Geo"] = "Yes"
    out["Same_Geo1"] = "Yes"
    return out


def yes_indicator(series: pd.Series | float) -> pd.Series | float:
    if isinstance(series, pd.Series):
        return series.astype(str).str.lower().str.contains("yes").astype(float)
    return series


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def build_medium_features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["one"] = 1.0
    x["incident_intensity"] = numeric(df, "log_input_flow")
    x["distance_attenuation"] = numeric(df, "log_AB_Dist")
    x["level_gap_U_minus_O"] = numeric(df, "level_gap_U_minus_O")
    x["U_level"] = numeric(df, "U_level_score")
    x["O_level"] = numeric(df, "O_level_score")
    x["same_province"] = yes_indicator(df["Same_Prov"]) if "Same_Prov" in df.columns else 0.0
    x["same_geo"] = yes_indicator(df["Same_Geo"]) if "Same_Geo" in df.columns else 0.0
    x["cross_region"] = 1.0 - x["same_geo"]

    for prefix, source in [("O", "A"), ("U", "B")]:
        x[f"log_{prefix}_popu"] = np.log1p(numeric(df, f"{source}_popu").clip(lower=0))
        x[f"log_{prefix}_gdp"] = np.log1p(numeric(df, f"{source}_gdp").clip(lower=0))
        x[f"log_{prefix}_house"] = np.log1p(numeric(df, f"{source}_house").clip(lower=0))
        x[f"{prefix}_tertiary"] = numeric(df, f"{source}_Tertiary_rate")
        x[f"{prefix}_hospital"] = numeric(df, f"{source}_hos")
        x[f"{prefix}_teacher"] = numeric(df, f"{source}_tea")
        x[f"{prefix}_museum"] = numeric(df, f"{source}_muse")
        x[f"{prefix}_center"] = numeric(df, f"{source}_cen")
        x[f"{prefix}_patent"] = numeric(df, f"{source}_avg_Num_patent")
        x[f"{prefix}_library"] = numeric(df, f"{source}_avg_num_Lib")
        x[f"{prefix}_aqi"] = numeric(df, f"{source}_aqi")

    x["gdp_gradient_U_minus_O"] = x["log_U_gdp"] - x["log_O_gdp"]
    x["population_gradient_U_minus_O"] = x["log_U_popu"] - x["log_O_popu"]
    x["house_pressure_U_minus_O"] = x["log_U_house"] - x["log_O_house"]
    x["center_gradient_U_minus_O"] = x["U_center"] - x["O_center"]
    x["tertiary_gradient_U_minus_O"] = x["U_tertiary"] - x["O_tertiary"]

    x["U_absorption_capacity"] = (
        x["log_U_gdp"] + x["U_tertiary"] + x["U_hospital"] + x["U_teacher"] + x["U_center"] - x["log_U_house"]
    )
    x["O_return_pull"] = x["log_O_gdp"] + x["O_center"] + x["same_province"] + x["same_geo"]
    x["U_gateway_openness"] = x["U_center"] + x["U_patent"] + x["U_tertiary"] + x["cross_region"]
    x["potential_gradient"] = x["U_absorption_capacity"] - x["O_return_pull"]
    x["U_absorption_weakness"] = -x["U_absorption_capacity"]
    return x


ABC_FEATURE_SETS = {
    "refraction": [
        "one",
        "U_gateway_openness",
        "potential_gradient",
        "level_gap_U_minus_O",
        "center_gradient_U_minus_O",
        "tertiary_gradient_U_minus_O",
        "distance_attenuation",
        "cross_region",
        "incident_intensity",
    ],
    "reflection": [
        "one",
        "O_return_pull",
        "same_province",
        "same_geo",
        "distance_attenuation",
        "house_pressure_U_minus_O",
        "level_gap_U_minus_O",
        "gdp_gradient_U_minus_O",
        "U_absorption_weakness",
        "incident_intensity",
    ],
    "absorption": [
        "one",
        "U_absorption_capacity",
        "potential_gradient",
        "U_level",
        "gdp_gradient_U_minus_O",
        "population_gradient_U_minus_O",
        "house_pressure_U_minus_O",
        "same_province",
        "incident_intensity",
    ],
}

DE_FEATURE_SETS = {
    "local_escape": [
        "one",
        "log_U_house",
        "U_aqi",
        "U_gateway_openness",
        "U_level",
        "incident_intensity",
    ],
    "local_retention": [
        "one",
        "U_absorption_capacity",
        "log_U_gdp",
        "U_tertiary",
        "U_level",
        "U_hospital",
        "U_teacher",
        "log_U_house",
        "incident_intensity",
    ],
}


DESIGN_TEXT = {
    "refraction": "第三城市折射响应：高校城市作为门户或跳板，将外地入射人才导向其他城市。",
    "reflection": "回乡反射响应：来源地牵引、区域记忆或高校城市吸收不足使人才回到家乡。",
    "absorption": "留城吸收响应：高校城市凭借经济、产业、教育和中心性将外地人才留在本地。",
    "local_escape": "本地逃逸响应：本地生从高校城市离开到其他城市。",
    "local_retention": "本地留存响应：本地生毕业后继续留在高校城市。",
}

FEATURE_TEXT = {
    "U_gateway_openness": "U 城市门户开放度：中心性、专利/创新、第三产业和跨区域通道性。",
    "potential_gradient": "U 相对 O 的综合吸引势差。",
    "level_gap_U_minus_O": "U 与 O 的城市等级差。",
    "center_gradient_U_minus_O": "U 与 O 的中心性差。",
    "tertiary_gradient_U_minus_O": "U 与 O 的第三产业比例差。",
    "distance_attenuation": "O-U 入射距离衰减。",
    "cross_region": "是否跨区域入射。",
    "incident_intensity": "入射光束强度 log(1 + flow)。",
    "O_return_pull": "来源城市回流牵引。",
    "same_province": "O 与 U 是否同省。",
    "same_geo": "O 与 U 是否同区域。",
    "house_pressure_U_minus_O": "U 相对 O 的房价压力差。",
    "gdp_gradient_U_minus_O": "U 与 O 的 GDP 梯度。",
    "U_absorption_weakness": "U 吸收能力不足项。",
    "U_absorption_capacity": "U 城市吸收能力：经济、产业、教育、医疗、中心性与房价压力。",
    "U_level": "U 城市等级。",
    "population_gradient_U_minus_O": "U 与 O 的人口规模梯度。",
    "log_U_house": "U 城市房价压力。",
    "U_aqi": "U 城市空气质量指标。",
    "log_U_gdp": "U 城市 GDP。",
    "U_tertiary": "U 城市第三产业比例。",
    "U_hospital": "U 城市医疗资源。",
    "U_teacher": "U 城市教育资源。",
    "one": "截距项。",
}


def fit_scaler(features: pd.DataFrame, feature_sets: dict[str, list[str]]) -> tuple[dict[str, float], dict[str, float]]:
    cols = sorted({c for cols in feature_sets.values() for c in cols if c != "one"})
    means, stds = {}, {}
    for col in cols:
        s = pd.to_numeric(features[col], errors="coerce")
        mean = float(s.mean()) if s.notna().any() else 0.0
        std = float(s.std(ddof=0)) if s.notna().any() else 1.0
        means[col] = mean if np.isfinite(mean) else 0.0
        stds[col] = std if np.isfinite(std) and std > EPS else 1.0
    return means, stds


def design_matrices(
    features: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    means: dict[str, float],
    stds: dict[str, float],
) -> dict[str, np.ndarray]:
    mats = {}
    for class_name, cols in feature_sets.items():
        parts = []
        for col in cols:
            if col == "one":
                parts.append(np.ones((len(features), 1), dtype=float))
            else:
                s = pd.to_numeric(features[col], errors="coerce").fillna(means.get(col, 0.0))
                z = (s - means.get(col, 0.0)) / stds.get(col, 1.0)
                parts.append(z.to_numpy(dtype=float).reshape(-1, 1))
        mats[class_name] = np.hstack(parts)
    return mats


def make_slices(feature_sets: dict[str, list[str]], class_names: list[str]) -> dict[str, slice]:
    slices = {}
    start = 0
    for class_name in class_names:
        stop = start + len(feature_sets[class_name])
        slices[class_name] = slice(start, stop)
        start = stop
    return slices


def softmax_scores(beta: np.ndarray, mats: dict[str, np.ndarray], slices: dict[str, slice], class_names: list[str]) -> np.ndarray:
    scores = np.column_stack([mats[name] @ beta[slices[name]] for name in class_names])
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), EPS, None)


def fit_structured_softmax(
    df: pd.DataFrame,
    class_names: list[str],
    feature_sets: dict[str, list[str]],
    weight_col: str,
    alpha_l2: float,
    max_iter: int,
) -> StructuredSoftmaxModel:
    features = build_medium_features(df)
    means, stds = fit_scaler(features, feature_sets)
    mats = design_matrices(features, feature_sets, means, stds)
    slices = make_slices(feature_sets, class_names)

    y_counts = df[class_names].to_numpy(dtype=float)
    y = y_counts / np.clip(y_counts.sum(axis=1, keepdims=True), EPS, None)
    weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n_params = max(s.stop for s in slices.values())

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        prob = softmax_scores(beta, mats, slices, class_names)
        ce = -np.sum(y * np.log(np.clip(prob, EPS, 1.0)), axis=1)
        loss = float(np.sum(weights * ce)) + 0.5 * alpha_l2 * float(np.sum(beta * beta))
        resid = (prob - y) * weights.reshape(-1, 1)
        grad = np.zeros_like(beta)
        for j, class_name in enumerate(class_names):
            grad[slices[class_name]] = mats[class_name].T @ resid[:, j]
        grad += alpha_l2 * beta
        return loss, grad

    result = minimize(lambda b: objective(b), np.zeros(n_params), jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
    if not result.success:
        print(f"Warning: optimizer did not fully converge: {result.message}")
    return StructuredSoftmaxModel(class_names, feature_sets, means, stds, result.x, slices, alpha_l2)


def predict_model(model: StructuredSoftmaxModel, df: pd.DataFrame) -> np.ndarray:
    features = build_medium_features(df)
    mats = design_matrices(features, model.feature_sets, model.means, model.stds)
    return softmax_scores(model.beta, mats, model.slices, model.class_names)


def add_predictions(df: pd.DataFrame, model: StructuredSoftmaxModel, weight_col: str, suffix: str = "") -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    probs = predict_model(model, out)
    prob_cols = []
    for i, class_name in enumerate(model.class_names):
        prob_col = f"P_{class_name}{suffix}"
        out[prob_col] = probs[:, i]
        out[f"pred_flow_{class_name}{suffix}"] = out[weight_col] * out[prob_col]
        prob_cols.append(prob_col)
    out[f"predicted_dominant{suffix}"] = out[prob_cols].idxmax(axis=1).str.replace("P_", "", regex=False).str.replace(suffix, "", regex=False)
    return out


def evaluate(name: str, df: pd.DataFrame, class_names: list[str], suffix: str, weight_col: str) -> list[dict[str, object]]:
    prob_cols = [f"P_{c}{suffix}" for c in class_names]
    weights = df[weight_col].to_numpy(dtype=float)
    actual_counts = df[class_names].to_numpy(dtype=float)
    actual = actual_counts / np.clip(actual_counts.sum(axis=1, keepdims=True), EPS, None)
    pred = df[prob_cols].to_numpy(dtype=float)
    ce = -np.sum(actual * np.log(np.clip(pred, EPS, 1.0)), axis=1)
    rows = [
        {"model": name, "metric": "weighted_fractional_cross_entropy", "value": float(np.average(ce, weights=weights))},
        {
            "model": name,
            "metric": "weighted_dominant_accuracy",
            "value": float(np.average(np.argmax(actual, axis=1) == np.argmax(pred, axis=1), weights=weights)),
        },
    ]
    for i, class_name in enumerate(class_names):
        rows.append({"model": name, "metric": f"{class_name}_weighted_rate_mae", "value": float(np.average(np.abs(actual[:, i] - pred[:, i]), weights=weights))})
        rows.append({"model": name, "metric": f"{class_name}_observed_share", "value": float(np.sum(actual[:, i] * weights) / weights.sum())})
        rows.append({"model": name, "metric": f"{class_name}_predicted_share", "value": float(np.sum(pred[:, i] * weights) / weights.sum())})
    return rows


def fit_predict(
    df: pd.DataFrame,
    class_names: list[str],
    feature_sets: dict[str, list[str]],
    weight_col: str,
    test_size: float,
    random_state: int,
    alpha_l2: float,
    max_iter: int,
) -> tuple[pd.DataFrame, pd.DataFrame, StructuredSoftmaxModel]:
    dominant = df[class_names].idxmax(axis=1)
    if len(df) >= 10 and dominant.nunique() > 1:
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            random_state=random_state,
            stratify=dominant if dominant.value_counts().min() >= 2 else None,
        )
    else:
        train_idx = np.arange(len(df))
        test_idx = np.arange(len(df))

    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    test_model = fit_structured_softmax(train, class_names, feature_sets, weight_col, alpha_l2, max_iter)
    test_pred = add_predictions(test, test_model, weight_col, "_test")

    final_model = fit_structured_softmax(df.reset_index(drop=True), class_names, feature_sets, weight_col, alpha_l2, max_iter)
    full_pred = add_predictions(df, final_model, weight_col)
    return full_pred, test_pred, final_model


def coefficient_table(model: StructuredSoftmaxModel) -> pd.DataFrame:
    rows = []
    for class_name in model.class_names:
        for feature, coef in zip(model.feature_sets[class_name], model.beta[model.slices[class_name]]):
            rows.append({"response": class_name, "feature": feature, "coefficient": coef})
    return pd.DataFrame(rows)


def design_table() -> pd.DataFrame:
    rows = []
    for module, feature_sets in [("ABC_foreign_response", ABC_FEATURE_SETS), ("DE_local_response", DE_FEATURE_SETS)]:
        for response, features in feature_sets.items():
            for feature in features:
                rows.append(
                    {
                        "module": module,
                        "response": response,
                        "response_meaning": DESIGN_TEXT.get(response, ""),
                        "feature": feature,
                        "feature_meaning": FEATURE_TEXT.get(feature, ""),
                    }
                )
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the main O-U probabilistic medium response model.")
    parser.add_argument("--main", required=True, help="Path to ReadytoRunModel_OUD_CityLevel.csv")
    parser.add_argument("--outdir", default="outputs/main_model", help="Output directory")
    parser.add_argument("--sample_rows", type=int, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--min_external_total", type=float, default=1.0)
    parser.add_argument("--min_local_total", type=float, default=1.0)
    parser.add_argument("--alpha_l2", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Reading data...")
    columns = needed_columns(existing_columns(args.main))
    raw = read_csv_safely(args.main, columns, args.sample_rows)
    df = add_basic_fields(raw)

    print("[2/7] Aggregating beams...")
    abc = aggregate_abc(df, args.min_external_total)
    de = aggregate_de(df, args.min_local_total)
    if abc.empty:
        raise ValueError("No ABC foreign O-U beams after filtering.")
    if de.empty:
        raise ValueError("No DE local beams after filtering.")

    print("[3/7] Fitting ABC foreign response...")
    abc_pred, abc_test, abc_model = fit_predict(
        abc, ABC_CLASSES, ABC_FEATURE_SETS, "F_external", args.test_size, args.random_state, args.alpha_l2, args.max_iter
    )

    print("[4/7] Fitting DE local response...")
    de_pred, de_test, de_model = fit_predict(
        de, DE_CLASSES, DE_FEATURE_SETS, "F_local", args.test_size, args.random_state, args.alpha_l2, args.max_iter
    )

    print("[5/7] Evaluating...")
    metrics = []
    metrics.extend(evaluate("ABC_foreign_medium_response", abc_test, ABC_CLASSES, "_test", "F_external"))
    metrics.extend(evaluate("DE_local_medium_response", de_test, DE_CLASSES, "_test", "F_local"))

    print("[6/7] Writing outputs...")
    save_csv(
        pd.DataFrame(
            [
                {"item": "raw_rows_after_clean", "value": len(df)},
                {"item": "abc_ou_beams", "value": len(abc)},
                {"item": "abc_external_flow", "value": abc["F_external"].sum()},
                {"item": "de_local_ucities", "value": len(de)},
                {"item": "de_local_flow", "value": de["F_local"].sum()},
                {"item": "alpha_l2", "value": args.alpha_l2},
                {"item": "output_dir", "value": str(outdir.resolve())},
            ]
        ),
        outdir / "00_input_summary.csv",
    )
    save_csv(abc_pred, outdir / "01_abc_foreign_medium_response_predicted.csv")
    save_csv(de_pred, outdir / "02_de_local_medium_response_predicted.csv")
    save_csv(pd.DataFrame(metrics), outdir / "03_model_metrics.csv")
    save_csv(design_table(), outdir / "04_model_design.csv")
    save_csv(coefficient_table(abc_model), outdir / "05_abc_coefficients.csv")
    save_csv(coefficient_table(de_model), outdir / "06_de_coefficients.csv")
    save_csv(abc_test, outdir / "07_abc_test_predictions.csv")
    save_csv(de_test, outdir / "08_de_test_predictions.csv")

    print("[7/7] Done.")
    print(f"Outputs: {outdir.resolve()}")


if __name__ == "__main__":
    main()
