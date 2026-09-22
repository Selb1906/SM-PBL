# -*- coding: utf-8 -*-
"""서울본부 2024-08-14 24시간 전력수요 예측 (트리 계열: 회귀트리·랜덤포레스트·XGBoost·LightGBM)
  1단계: 전력사용량 파일만        2단계: + 기상(ASOS 서울) 파일        3단계: 2024년 8월 전체(31일) 하루 전 예측
결과: results/*.csv, figures/*.png.  README 의 표는 이 스크립트 출력으로 채움.
실행: python forecast.py   (약 7분: 검증 1분 + 튜닝 3분 + 8월 rolling 3분)"""
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
DATA = BASE / "data"                                        # 데이터 CSV 두 개
OUT_R, OUT_F = BASE / "results", BASE / "figures"
OUT_R.mkdir(exist_ok=True); OUT_F.mkdir(exist_ok=True)

REGION = "서울본부"
TARGET = pd.Timestamp("2024-08-14")
VAL_START, VAL_END = pd.Timestamp("2024-07-17"), pd.Timestamp("2024-08-13")   # 검증 4주 (튜닝·모델 선택용)
AUG = pd.date_range("2024-08-01", "2024-08-31", freq="D")
SEED = 0

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, RED, GREEN, SURFACE = "#2a78d6", "#eb6834", "#e34948", "#3a9a68", "#fcfcfb"
plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "font.size": 11,
})

# 공휴일 (대체공휴일·선거일·임시공휴일 포함)
HOLIDAYS = pd.to_datetime([
    "2022-01-01", "2022-01-31", "2022-02-01", "2022-02-02", "2022-03-01", "2022-03-09", "2022-05-05", "2022-05-08", "2022-06-01", "2022-06-06",
    "2022-08-15", "2022-09-09", "2022-09-10", "2022-09-11", "2022-09-12", "2022-10-03", "2022-10-09", "2022-10-10", "2022-12-25",
    "2023-01-01", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24", "2023-03-01", "2023-05-05", "2023-05-27", "2023-05-29", "2023-06-06",
    "2023-08-15", "2023-09-28", "2023-09-29", "2023-09-30", "2023-10-02", "2023-10-03", "2023-10-09", "2023-12-25",
    "2024-01-01", "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12", "2024-03-01", "2024-04-10", "2024-05-06", "2024-05-15", "2024-06-06",
    "2024-08-15", "2024-09-16", "2024-09-17", "2024-09-18", "2024-10-01", "2024-10-03", "2024-10-09", "2024-12-25",
])


# ═══ 1. 데이터 ══════════════════════════════════════════════════════════════
def load_data():
    df = pd.read_csv(DATA / "한국전력공사_전국 시간별 전력사용량_20241231.csv", encoding="utf-8-sig")
    d = df[df["본부명"] == REGION].copy()
    # 기준시 h = h시에 끝나는 한 시간 (1~24) → 시각 = 날짜 + h 시간 (24시는 다음날 00:00)
    d["ts"] = pd.to_datetime(d["기준일자"]) + pd.to_timedelta(d["기준시"], unit="h")
    d = d.sort_values("ts").set_index("ts")[["전력사용량"]].rename(columns={"전력사용량": "load"})
    assert (d.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all(), "시간이 빠짐없이 이어져야 함"
    w = pd.read_csv(DATA / "ASOS_108_20210101_20250101.csv", encoding="utf-8-sig")
    w["ts"] = pd.to_datetime(w["date"])
    w = w.set_index("ts")[["기온", "습도", "일사", "전운량"]].rename(columns={"기온": "temp", "습도": "humid", "일사": "solar", "전운량": "cloud"})
    w = w.reindex(d.index).interpolate(limit=3)             # 짧은 결측만 보간
    w["solar"] = w["solar"].fillna(0.0)                      # 일사 결측 = 밤 → 0
    return d.join(w)


def add_features(d):
    """모든 특징은 '예측일 전날(D-1) 24시까지 아는 값'으로만 만든다 → 부하 lag ≥ 24h.
    기상은 예측일 당일 관측값을 '정확한 예보'로 가정해 사용 (실무에서는 기상예보로 대체)."""
    f = pd.DataFrame(index=d.index)
    day = d.index - pd.Timedelta(hours=1)                    # 이 시각이 속한 '날' (24시 = 그날의 마지막 시간)
    f["hour"] = day.hour + 1
    f["dow"] = day.dayofweek
    f["weekend"] = (f["dow"] >= 5).astype(int)
    f["holiday"] = pd.Series(day.normalize(), index=d.index).isin(HOLIDAYS).astype(int).values
    f["offday"] = ((f["weekend"] == 1) | (f["holiday"] == 1)).astype(int)
    f["doy"] = day.dayofyear
    L = d["load"]
    f["lag24"] = L.shift(24)                                 # 전날 같은 시각
    f["lag48"] = L.shift(48)
    f["lag168"] = L.shift(168)                               # 지난주 같은 요일 같은 시각
    f["lag_wk_mean"] = sum(L.shift(24 * k) for k in range(1, 8)) / 7   # 지난 7일 같은 시각 평균
    f["lag24_daymean"] = L.shift(24).rolling(24).mean()      # 전날 하루 평균 (요즘 수준)
    f["lag24_daymax"] = L.shift(24).rolling(24).max()
    # 기상 (예측일 당일 = 예보 가정)
    f["temp"] = d["temp"]; f["humid"] = d["humid"]; f["solar"] = d["solar"]; f["cloud"] = d["cloud"]
    f["cdh"] = (d["temp"] - 24).clip(lower=0)                # 냉방도 (24℃ 초과분)
    f["temp_daymax"] = d["temp"].groupby(day.normalize()).transform("max")
    f["temp_lag24"] = d["temp"].shift(24)
    f["y"] = L
    return f


FEAT_CAL = ["hour", "dow", "weekend", "holiday", "offday", "doy"]
FEAT_LAG = ["lag24", "lag48", "lag168", "lag_wk_mean", "lag24_daymean", "lag24_daymax"]
FEAT_WX = ["temp", "humid", "solar", "cloud", "cdh", "temp_daymax", "temp_lag24"]
FEATSETS = {"1단계: 전력 파일만": FEAT_CAL + FEAT_LAG, "2단계: + 기상": FEAT_CAL + FEAT_LAG + FEAT_WX}

raw = load_data()
F = add_features(raw).dropna()
day_of = (F.index - pd.Timedelta(hours=1)).normalize()


def rows_of_day(dt):
    return F[day_of == dt]


def metrics(y, yhat):
    e = np.asarray(y) - np.asarray(yhat)
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAPE": float(np.mean(np.abs(e) / np.asarray(y)) * 100)}


# ═══ 2. 후보 모델 ═══════════════════════════════════════════════════════════
class DiffModel:
    """트리는 학습 범위 밖 값을 못 만든다(외삽 불가) → 'y' 대신 'y - 지난 7일 같은 시각 평균'(차이)을 예측하고 나중에 더해 줌."""
    def __init__(self, base, ref="lag_wk_mean"):
        self.base, self.ref = base, ref

    def fit(self, X, y):
        self.base.fit(X, y - X[self.ref]); return self

    def predict(self, X):
        return self.base.predict(X) + X[self.ref].values

    def get_params(self, deep=False):
        return {"base": self.base, "ref": self.ref}


def build(name, params=None):
    """이름 → 모델 객체. params 가 있으면 그 하이퍼파라미터로 만듦 (튜닝용). 트리 계열만 후보."""
    p_ = params or {}
    if name == "회귀트리":
        return DecisionTreeRegressor(random_state=SEED, **{"min_samples_leaf": 20, **p_})
    rf = lambda: RandomForestRegressor(n_jobs=-1, random_state=SEED, **{"n_estimators": 200, "min_samples_leaf": 3, "max_features": 0.6, **p_})
    xgb = lambda: XGBRegressor(tree_method="hist", subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=0,
                               **{"n_estimators": 500, "learning_rate": 0.05, "max_depth": 5, **p_})
    lgb = lambda: LGBMRegressor(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1,
                                **{"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, **p_})
    return {"랜덤포레스트": rf, "XGBoost": xgb, "LightGBM": lgb,
            "랜덤포레스트(차이 예측)": lambda: DiffModel(rf()), "XGBoost(차이 예측)": lambda: DiffModel(xgb()), "LightGBM(차이 예측)": lambda: DiffModel(lgb())}[name]()


MODEL_NAMES = ["회귀트리", "랜덤포레스트", "XGBoost", "LightGBM", "랜덤포레스트(차이 예측)", "XGBoost(차이 예측)", "LightGBM(차이 예측)"]
GRIDS = {
    "회귀트리": [dict(min_samples_leaf=l, max_depth=d) for l in (5, 20, 50) for d in (4, 6, 8, None)],
    "랜덤포레스트": [dict(n_estimators=n, min_samples_leaf=l, max_features=mf) for n in (200, 400) for l in (1, 3, 8) for mf in (0.4, 0.6, 1.0)],
    "XGBoost": [dict(n_estimators=n, learning_rate=lr, max_depth=d) for n in (300, 600, 1000) for lr in (0.02, 0.05, 0.1) for d in (3, 5, 7)],
    "LightGBM": [dict(n_estimators=n, learning_rate=lr, num_leaves=nl) for n in (300, 600, 1000) for lr in (0.02, 0.05, 0.1) for nl in (15, 31, 63)],
}
for k in ("랜덤포레스트", "XGBoost", "LightGBM"):
    GRIDS[k + "(차이 예측)"] = GRIDS[k]


def make_models():
    return {n: build(n) for n in MODEL_NAMES}


def fit_predict(model, feats, train_end, pred_days, window_days=None):
    """train_end(포함) 까지의 행으로 학습 → pred_days 의 24시간씩 예측. window_days 로 학습 기간 제한."""
    tr = F[day_of <= train_end]
    if window_days:
        tr = tr[day_of[day_of <= train_end] > train_end - pd.Timedelta(days=window_days)]
    model.fit(tr[feats], tr["y"])
    out = []
    for dt in pred_days:
        te = rows_of_day(dt)
        out.append(pd.DataFrame({"day": dt, "hour": te["hour"].values, "y": te["y"].values, "yhat": model.predict(te[feats])}))
    return pd.concat(out, ignore_index=True)


def baselines(pred_days):
    rows = []
    for dt in pred_days:
        te = rows_of_day(dt)
        rows.append(pd.DataFrame({"day": dt, "hour": te["hour"].values, "y": te["y"].values,
                                  "전날 그대로": te["lag24"].values, "지난주 같은 요일": te["lag168"].values, "지난 7일 같은 시각 평균": te["lag_wk_mean"].values}))
    return pd.concat(rows, ignore_index=True)


# ═══ 3. 검증(2024-07-17~08-13, 28일)으로 모델·특징·학습기간 고르기 ═══════════
t0 = time.time()
val_days = pd.date_range(VAL_START, VAL_END, freq="D")
val_rows = []
bl = baselines(val_days)
for name in ["전날 그대로", "지난주 같은 요일", "지난 7일 같은 시각 평균"]:
    val_rows.append({"특징": "기준선", "모델": name, "학습기간": "-", **metrics(bl["y"], bl[name])})
for fs_name, feats in FEATSETS.items():                      # 비교용 선형회귀 (트리가 아니므로 최종 후보에서는 제외)
    pr = fit_predict(Ridge(alpha=1.0), feats, VAL_START - pd.Timedelta(days=1), val_days)
    val_rows.append({"특징": "기준선", "모델": f"선형회귀 ({fs_name[:3]})", "학습기간": "전체(2022~)", **metrics(pr["y"], pr["yhat"])})
for fs_name, feats in FEATSETS.items():
    for m_name, model in make_models().items():
        # 검증 28일은 하루 전 예측을 흉내내되, 속도를 위해 검증 시작 전날까지 한 번만 학습 (lag 특징은 실제값 사용)
        pr = fit_predict(model, feats, VAL_START - pd.Timedelta(days=1), val_days)
        val_rows.append({"특징": fs_name, "모델": m_name, "학습기간": "전체(2022~)", **metrics(pr["y"], pr["yhat"])})
val = pd.DataFrame(val_rows)
print(val.round(1).to_string(index=False)); print(f"[검증 1차 {time.time() - t0:.0f}s]")

# 학습기간 튜닝 (특징 2단계 기준, 상위 2개 모델)
top2 = val[val["특징"] == "2단계: + 기상"].nsmallest(3, "RMSE")["모델"].tolist()
win_rows = []
for m_name in top2:
    for win_name, win in [("최근 8주", 56), ("최근 1년", 365), ("전체(2022~)", None)]:
        pr = fit_predict(build(m_name), FEATSETS["2단계: + 기상"], VAL_START - pd.Timedelta(days=1), val_days, window_days=win)
        win_rows.append({"모델": m_name, "학습기간": win_name, **metrics(pr["y"], pr["yhat"])})
win = pd.DataFrame(win_rows)
print(win.round(1).to_string(index=False))

# 하이퍼파라미터 튜닝 (검증 RMSE 최소인 모델 계열에 대해 작은 격자)
best_row = win.nsmallest(1, "RMSE").iloc[0]
BEST_MODEL, BEST_WIN = best_row["모델"], {"최근 8주": 56, "최근 1년": 365, "전체(2022~)": None}[best_row["학습기간"]]
grid_rows = []
grid = GRIDS[BEST_MODEL]
mk = lambda p_: build(BEST_MODEL, p_)
for p_ in grid:
    pr = fit_predict(mk(p_), FEATSETS["2단계: + 기상"], VAL_START - pd.Timedelta(days=1), val_days, window_days=BEST_WIN)
    grid_rows.append({**{k: str(v) for k, v in p_.items()}, **metrics(pr["y"], pr["yhat"])})
grid_df = pd.DataFrame(grid_rows).sort_values("RMSE")
BEST_PARAMS = grid[int(grid_df.index[0])]
print(f"선택: {BEST_MODEL}, 학습기간 {best_row['학습기간']}, 파라미터 {BEST_PARAMS}, 검증 RMSE {grid_df.iloc[0]['RMSE']:.1f}")
print(f"[튜닝 {time.time() - t0:.0f}s]")
val.to_csv(OUT_R / "00_검증_모델비교.csv", index=False, encoding="utf-8-sig", float_format="%.2f")
win.to_csv(OUT_R / "00_검증_학습기간.csv", index=False, encoding="utf-8-sig", float_format="%.2f")
grid_df.to_csv(OUT_R / "00_검증_하이퍼파라미터.csv", index=False, encoding="utf-8-sig", float_format="%.2f")

# ═══ 4. 1·2단계: 2024-08-14 예측 (전날 08-13 까지 학습) ═══════════════════
final_model = lambda: mk(BEST_PARAMS)
stage = {}
bl14 = baselines([TARGET])
res14 = [{"단계": "기준선", "모델": n, **metrics(bl14["y"], bl14[n])} for n in ["전날 그대로", "지난주 같은 요일", "지난 7일 같은 시각 평균"]]
for fs_name, feats in FEATSETS.items():
    pr = fit_predict(Ridge(alpha=1.0), feats, TARGET - pd.Timedelta(days=1), [TARGET])
    res14.append({"단계": "기준선", "모델": f"선형회귀 ({fs_name[:3]})", **metrics(pr["y"], pr["yhat"])})
for fs_name, feats in FEATSETS.items():
    for m_name, model in {**make_models(), f"{BEST_MODEL} (튜닝)": final_model()}.items():
        pr = fit_predict(model, feats, TARGET - pd.Timedelta(days=1), [TARGET], window_days=BEST_WIN if "튜닝" in m_name else None)
        res14.append({"단계": fs_name, "모델": m_name, **metrics(pr["y"], pr["yhat"])})
        stage[(fs_name, m_name)] = pr
res14 = pd.DataFrame(res14)
print(res14.round(1).to_string(index=False))
res14.to_csv(OUT_R / "01_02_0814_결과.csv", index=False, encoding="utf-8-sig", float_format="%.2f")

# 그림: 08-14 예측 곡선 (1단계 vs 2단계, 튜닝 모델)
t = np.arange(1, 25)
p1 = stage[("1단계: 전력 파일만", f"{BEST_MODEL} (튜닝)")]; p2 = stage[("2단계: + 기상", f"{BEST_MODEL} (튜닝)")]
m1, m2, mb = metrics(p1["y"], p1["yhat"]), metrics(p2["y"], p2["yhat"]), metrics(bl14["y"], bl14["전날 그대로"])
fig, (ax, axw) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12})
ax.plot(t, bl14["전날 그대로"], color=MUTED, lw=1.6, ls=(0, (4, 2)), label=f"기준선: 전날 그대로  RMSE {mb['RMSE']:.0f} · MAPE {mb['MAPE']:.1f}%")
ax.plot(t, p1["yhat"], color=ORANGE, lw=2.2, label=f"1단계 전력 파일만 ({BEST_MODEL})  RMSE {m1['RMSE']:.0f} · MAPE {m1['MAPE']:.1f}%")
ax.plot(t, p2["yhat"], color=BLUE, lw=2.6, label=f"2단계 + 기상 ({BEST_MODEL})  RMSE {m2['RMSE']:.0f} · MAPE {m2['MAPE']:.1f}%")
ax.plot(t, p1["y"], color=INK, lw=1.6, marker="o", ms=4.5, mfc=SURFACE, mew=1.4, label="실제값 2024-08-14 (수)", zorder=6)
ax.set_ylabel("전력사용량 (MWh)"); ax.legend(loc="upper left", fontsize=9.5); ax.grid(axis="x", visible=False)
ax.set_title(f"2024-08-14 24시간 예측 — {REGION}, 전날(08-13)까지 학습", loc="left", fontsize=14, fontweight="bold")
te14 = rows_of_day(TARGET)
axw.plot(t, te14["temp"], color=RED, lw=2, marker="o", ms=3.5, label="기온 (℃)"); axw.set_ylabel("기온 (℃)")
ax2 = axw.twinx(); ax2.bar(t, te14["solar"], 0.6, color="#f6c9b5", zorder=1, label="일사 (MJ/m²)"); ax2.set_ylabel("일사 (MJ/m²)"); ax2.grid(False)
axw.set_xticks(t[::2]); axw.set_xlim(0.5, 24.5); axw.set_xlabel("시간 (시)"); axw.grid(axis="x", visible=False)
axw.set_title("예측일 기상 (관측값을 '정확한 예보'로 가정)", loc="left", fontsize=11.5, fontweight="bold")
h1, l1 = axw.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); axw.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9.5)
fig.savefig(OUT_F / "01_0814_예측곡선.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 그림: 특징 중요도 (2단계 튜닝 모델)
mdl = final_model(); tr = F[day_of <= TARGET - pd.Timedelta(days=1)]
if BEST_WIN:
    tr = tr[day_of[day_of <= TARGET - pd.Timedelta(days=1)] > TARGET - pd.Timedelta(days=1 + BEST_WIN)]
mdl.fit(tr[FEATSETS["2단계: + 기상"]], tr["y"])
# 순열 중요도: 검증 28일에서 특징 하나를 무작위로 섞었을 때 RMSE 가 얼마나 나빠지나 (모델 종류와 무관하게 계산 가능)
vrows = F[(day_of >= VAL_START) & (day_of <= VAL_END)]
rng = np.random.default_rng(SEED)
base_rmse = metrics(vrows["y"], mdl.predict(vrows[FEATSETS["2단계: + 기상"]]))["RMSE"]
imp = {}
for k in FEATSETS["2단계: + 기상"]:
    Xp = vrows[FEATSETS["2단계: + 기상"]].copy(); Xp[k] = rng.permutation(Xp[k].values)
    imp[k] = metrics(vrows["y"], mdl.predict(Xp))["RMSE"] - base_rmse
imp = pd.Series(imp).sort_values()
fig, ax = plt.subplots(figsize=(9, 6.2))
cols = [GREEN if k in FEAT_WX else (BLUE if k in FEAT_LAG else MUTED) for k in imp.index]
ax.barh(imp.index, imp.values, 0.6, color=cols, zorder=2)
for i, v in enumerate(imp.values):
    ax.text(v + imp.max() * 0.01, i, f"{v:.3f}", va="center", fontsize=9.5, color=INK2)
ax.set_xlim(min(0, imp.min() * 1.2), imp.max() * 1.15); ax.axvline(0, color=AXIS, lw=1); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
ax.set_xlabel(f"순열 중요도: 그 특징을 섞었을 때 검증 RMSE 증가 (MWh, 기준 {base_rmse:.0f})")
ax.plot([], [], color=MUTED, lw=8, label="달력"); ax.plot([], [], color=BLUE, lw=8, label="과거 부하 (lag)"); ax.plot([], [], color=GREEN, lw=8, label="기상"); ax.legend(loc="lower right")
ax.set_title(f"특징 중요도 (순열) — {BEST_MODEL} (2단계 특징, 검증 28일)", loc="left", fontsize=13, fontweight="bold")
fig.savefig(OUT_F / "02_특징중요도.png", dpi=200, bbox_inches="tight"); plt.close(fig)
imp.sort_values(ascending=False).to_csv(OUT_R / "02_특징중요도.csv", encoding="utf-8-sig", float_format="%.2f")

# ═══ 5. 3단계: 8월 전체 — 매일 전날까지 다시 학습해 다음 날 예측 (rolling) ═════
t0 = time.time()
aug_rows, aug_pred = [], []
blA = baselines(AUG)
for fs_name, feats in FEATSETS.items():
    parts = []
    for dt in AUG:
        pr = fit_predict(final_model(), feats, dt - pd.Timedelta(days=1), [dt], window_days=BEST_WIN)
        parts.append(pr)
    pr = pd.concat(parts, ignore_index=True); pr["특징"] = fs_name
    aug_pred.append(pr)
aug_pred = pd.concat(aug_pred, ignore_index=True)
print(f"[8월 rolling {time.time() - t0:.0f}s]")
resA = [{"모델": n, **metrics(blA["y"], blA[n])} for n in ["전날 그대로", "지난주 같은 요일", "지난 7일 같은 시각 평균"]]
for fs_name in FEATSETS:
    q = aug_pred[aug_pred["특징"] == fs_name]
    resA.append({"모델": f"{fs_name} · {BEST_MODEL}(튜닝, 매일 재학습)", **metrics(q["y"], q["yhat"])})
resA = pd.DataFrame(resA)
print(resA.round(1).to_string(index=False))
resA.to_csv(OUT_R / "03_8월전체_결과.csv", index=False, encoding="utf-8-sig", float_format="%.2f")

daily = []
for dt in AUG:
    q1 = aug_pred[(aug_pred["특징"] == "1단계: 전력 파일만") & (aug_pred["day"] == dt)]
    q2 = aug_pred[(aug_pred["특징"] == "2단계: + 기상") & (aug_pred["day"] == dt)]
    b = blA[blA["day"] == dt]
    daily.append({"day": dt.strftime("%m-%d"), "요일": "월화수목금토일"[dt.dayofweek], "공휴일": int(dt in HOLIDAYS),
                  "기준선 RMSE": metrics(b["y"], b["전날 그대로"])["RMSE"], "1단계 RMSE": metrics(q1["y"], q1["yhat"])["RMSE"], "2단계 RMSE": metrics(q2["y"], q2["yhat"])["RMSE"],
                  "기준선 MAPE": metrics(b["y"], b["전날 그대로"])["MAPE"], "1단계 MAPE": metrics(q1["y"], q1["yhat"])["MAPE"], "2단계 MAPE": metrics(q2["y"], q2["yhat"])["MAPE"]})
daily = pd.DataFrame(daily)
daily.to_csv(OUT_R / "03_8월_일별_오차.csv", index=False, encoding="utf-8-sig", float_format="%.2f")
aug_pred.to_csv(OUT_R / "03_8월_시간별_예측.csv", index=False, encoding="utf-8-sig", float_format="%.1f")

# 그림: 8월 전체 곡선 + 일별 MAPE
fig, (ax, axd) = plt.subplots(2, 1, figsize=(15, 8.5), gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.32})
q2 = aug_pred[aug_pred["특징"] == "2단계: + 기상"]; q1 = aug_pred[aug_pred["특징"] == "1단계: 전력 파일만"]
xs = q2["day"] + pd.to_timedelta(q2["hour"], unit="h")
ax.plot(xs, q2["y"], color=INK, lw=1.2, label="실제값")
ax.plot(xs, q1["yhat"], color=ORANGE, lw=1.0, alpha=0.9, label=f"1단계 전력 파일만  RMSE {resA.iloc[3]['RMSE']:.0f} · MAPE {resA.iloc[3]['MAPE']:.1f}%")
ax.plot(xs, q2["yhat"], color=BLUE, lw=1.2, label=f"2단계 + 기상  RMSE {resA.iloc[4]['RMSE']:.0f} · MAPE {resA.iloc[4]['MAPE']:.1f}%")
for dt in AUG:
    if dt.dayofweek >= 5 or dt in HOLIDAYS:
        ax.axvspan(dt, dt + pd.Timedelta(days=1), color="#fde9df", zorder=0)
ax.set_ylabel("전력사용량 (MWh)"); ax.legend(loc="upper left", ncol=3, fontsize=9.5); ax.grid(axis="x", visible=False)
ax.set_title(f"2024년 8월 전체 하루 전 예측 — {REGION}, 매일 전날까지 재학습 ({BEST_MODEL}) · 음영 = 주말·공휴일", loc="left", fontsize=13.5, fontweight="bold")
xd = np.arange(len(daily))
axd.bar(xd - 0.27, daily["기준선 MAPE"], 0.27, color=MUTED, label="기준선 전날 그대로")
axd.bar(xd, daily["1단계 MAPE"], 0.27, color=ORANGE, label="1단계")
axd.bar(xd + 0.27, daily["2단계 MAPE"], 0.27, color=BLUE, label="2단계")
axd.set_xticks(xd); axd.set_xticklabels([f"{d}\n{w}" for d, w in zip(daily["day"], daily["요일"])], fontsize=8.5)
axd.set_ylabel("MAPE (%)"); axd.grid(axis="x", visible=False); axd.legend(loc="upper right", ncol=3, fontsize=9.5)
axd.set_title("일별 MAPE — 어떤 날이 어려운가? (공휴일 08-15, 주말, 날씨가 급변한 날)", loc="left", fontsize=12, fontweight="bold")
fig.savefig(OUT_F / "03_8월전체_예측.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 그림: 검증 결과 요약 (모델 × 특징)
piv = val[val["특징"] != "기준선"].pivot(index="모델", columns="특징", values="RMSE").loc[MODEL_NAMES]
fig, ax = plt.subplots(figsize=(12.5, 5.2))
xp = np.arange(len(piv))
ax.bar(xp - 0.2, piv.iloc[:, 0], 0.4, color=ORANGE, label=piv.columns[0]); ax.bar(xp + 0.2, piv.iloc[:, 1], 0.4, color=BLUE, label=piv.columns[1])
for i in range(len(piv)):
    ax.text(xp[i] - 0.2, piv.iloc[i, 0] + 3, f"{piv.iloc[i, 0]:.0f}", ha="center", fontsize=9.5, color=INK2); ax.text(xp[i] + 0.2, piv.iloc[i, 1] + 3, f"{piv.iloc[i, 1]:.0f}", ha="center", fontsize=9.5, color=INK2)
bl_r = val[val["모델"] == "전날 그대로"]["RMSE"].iloc[0]; lr_r = val[val["모델"] == "선형회귀 (2단계)"]["RMSE"].iloc[0]
ax.axhline(bl_r, color=MUTED, lw=1.2, ls=(0, (4, 2))); ax.text(len(piv) - 0.5, bl_r + 3, f"기준선 전날 그대로 {bl_r:.0f}", ha="right", fontsize=9.5, color=INK2)
ax.axhline(lr_r, color=RED, lw=1.2, ls=(0, (4, 2))); ax.text(len(piv) - 0.5, lr_r + 3, f"비교용 선형회귀 (2단계 특징) {lr_r:.0f}", ha="right", fontsize=9.5, color=RED)
ax.set_xticks(xp); ax.set_xticklabels([n.replace("(차이 예측)", "\n(차이 예측)") for n in piv.index], fontsize=9.5); ax.set_ylabel("검증 RMSE (MWh)"); ax.grid(axis="x", visible=False); ax.legend(loc="upper right")
ax.set_title(f"모델 선택 근거 — 검증 기간 {VAL_START:%m-%d}~{VAL_END:%m-%d} (28일, 예측일 이전)", loc="left", fontsize=13, fontweight="bold")
fig.savefig(OUT_F / "00_검증_모델비교.png", dpi=200, bbox_inches="tight"); plt.close(fig)

with open(OUT_R / "선택요약.txt", "w", encoding="utf-8") as fh:
    fh.write(f"모델: {BEST_MODEL}\n학습기간: {best_row['학습기간']}\n파라미터: {BEST_PARAMS}\n검증 RMSE: {grid_df.iloc[0]['RMSE']:.2f}\n")
print("done")
