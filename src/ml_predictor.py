import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score

from feature_extractor import FEATURE_NAMES, extract_features, to_vector

RESULTS_FILE = "./data/results_concat.csv"
MODEL_FILE = "./data/optik_models.joblib"
FIG_DIR = "./data/figures"
V_MAX = 1e10               # must match the data maker
POA_SIMD_SPEEDUP = 8.0     # assumed SPOA SIMD speedup; no-SIMD POA time = measured time x this
SEED = 42

METHODS = ["DP", "POA", "SECTION", "OPTIK"]
LABELS = {"DP": "Full DP", "POA": "POA (no-SIMD est.)", "SECTION": "LCSk++ sectioning", "OPTIK": "OPTIK (ML)"}
COLORS = {"DP": "#2a78d6", "POA": "#e07b00", "SECTION": "#3a9a3a", "OPTIK": "#7b4fc9"}
SIZES = [20, 100, 500, 1000]
BIN_LABELS = {20: "≤20", 100: "21–100", 500: "101–500", 1000: "501–1000"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "savefig.dpi": 300})


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_results(path=RESULTS_FILE):
    """Load the data maker output, drop failed sets, and estimate no-SIMD POA time."""
    df = pd.read_csv(path)
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]
    df = df.copy()
    df["set_id"] = df["set_id"].astype(str)
    print(f"Loaded {len(df)} sets")
    print(f"  with DP optimum:        {df['dp_score'].notna().sum()}")
    print(f"  with sectioning result: {df['section_score'].notna().sum()}")
    if df["section_frac"].notna().any():
        exact = (df["section_frac"] >= 1 - 1e-9).mean()
        print(f"  sectioning = optimum in {exact:.1%} of sets with DP")

    # POA time without SIMD (estimated): measured SPOA time x assumed SIMD speedup
    df["time_poa_measured"] = df["time_poa"]
    df["time_poa"] = df["time_poa"] * POA_SIMD_SPEEDUP
    print(f"  POA times scaled by x{POA_SIMD_SPEEDUP} to estimate no-SIMD runtime")
    return df


def split_by_set(df, seed=SEED):
    """70/15/15 split by set_id, so regions from one set stay in one split."""
    gss = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, rest_idx = next(gss.split(df, groups=df["set_id"]))
    rest = df.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss2.split(rest, groups=rest["set_id"]))
    return df.iloc[train_idx], rest.iloc[val_idx], rest.iloc[test_idx]


# ---------------------------------------------------------------------------
# Deterministic time formulas (fit one constant each)
# ---------------------------------------------------------------------------

def fit_time_constant(work, seconds):
    """Fit seconds ≈ c * work in log space; returns c."""
    work, seconds = np.asarray(work, float), np.asarray(seconds, float)
    ok = np.isfinite(work) & np.isfinite(seconds) & (work > 0) & (seconds > 0)
    return float(np.exp(np.mean(np.log(seconds[ok]) - np.log(work[ok]))))


def dp_work(volume):
    return 7 * np.asarray(volume, float)             # 7 moves per cell for N = 3


def poa_work(mean_len, n=3):
    return n * np.asarray(mean_len, float) ** 2      # each sequence aligned to a graph of ~L nodes


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_model():
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
        early_stopping=True, validation_fraction=0.1, random_state=SEED)


def fit_and_report(name, target, train_df, val_df, test_df, transform=None):
    """Train one regressor on rows where the target exists; print MAE vs a mean baseline."""
    def xy(d):
        y = d[target].astype(float)
        if transform is not None:
            y = transform(y)
        ok = np.isfinite(y)
        return d.loc[ok, FEATURE_NAMES].values, y[ok].values

    X_tr, y_tr = xy(train_df)
    X_va, y_va = xy(val_df)
    X_te, y_te = xy(test_df)
    model = make_model().fit(X_tr, y_tr)

    base = np.full_like(y_te, y_tr.mean())
    print(f"{name:>18}  n={len(y_tr):6d}  val MAE {mean_absolute_error(y_va, model.predict(X_va)):.4f}"
          f"  test MAE {mean_absolute_error(y_te, model.predict(X_te)):.4f}"
          f"  (mean baseline {mean_absolute_error(y_te, base):.4f})")
    return model


def train(df, path=MODEL_FILE):
    """Train all models and save them, with everything needed to use them later."""
    train_df, val_df, test_df = split_by_set(df)
    print(f"Split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test sets\n")

    models = {}

    # 1-2: scores as a fraction of the optimum (only sets where DP ran)
    models["score_poa"] = fit_and_report("POA score", "score_poa", train_df, val_df, test_df)
    models["score_section"] = fit_and_report("Section score", "score_section", train_df, val_df, test_df)

    # 3: sectioning time (parallel), in log10 seconds
    models["log_time_section"] = fit_and_report(
        "Section time (log)", "time_section", train_df, val_df, test_df,
        transform=lambda t: np.log10(t.where(t > 0)))

    # 4-5: time constants for DP and POA (POA uses the no-SIMD estimate)
    models["c_dp"] = fit_time_constant(dp_work(train_df["volume"]), train_df["time_dp"])
    models["c_poa"] = fit_time_constant(poa_work(train_df["mean_len"]), train_df["time_poa"])
    print(f"\nc_dp  = {models['c_dp']:.3e} s per cell-move")
    print(f"c_poa = {models['c_poa']:.3e} s per unit (N * L^2), no-SIMD estimate")

    # Everything needed to use the model later
    models["feature_names"] = list(FEATURE_NAMES)
    models["v_max"] = V_MAX
    models["poa_simd_speedup"] = POA_SIMD_SPEEDUP
    models["test_set_ids"] = sorted(test_df["set_id"].unique())

    joblib.dump(models, path)
    print(f"Saved models to {path}\n")
    return models


# ---------------------------------------------------------------------------
# Loading the saved model
# ---------------------------------------------------------------------------

_MODELS = None


def load_models(path=MODEL_FILE):
    """Load the saved models once and reuse them for every prediction."""
    global _MODELS
    if _MODELS is None:
        _MODELS = joblib.load(path)
        if _MODELS["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("Saved model was trained with different features; retrain it.")
        # Rescale the POA time constant if the model was saved with a different SIMD factor
        saved = _MODELS.get("poa_simd_speedup", 1.0)
        if saved != POA_SIMD_SPEEDUP:
            _MODELS["c_poa"] *= POA_SIMD_SPEEDUP / saved
            _MODELS["poa_simd_speedup"] = POA_SIMD_SPEEDUP
            print(f"Rescaled POA time constant from x{saved} to x{POA_SIMD_SPEEDUP} SIMD factor")
        print(f"Loaded models from {path}")
    return _MODELS


def test_split(df, models):
    """The test sets saved with the model."""
    return df[df["set_id"].isin(set(models["test_set_ids"]))]


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def predict_from_features(models, feats, volume, mean_len, n_anchors):
    """Predicted (score, time, allowed) for each strategy, from a feature vector."""
    v = np.asarray(feats, dtype=np.float32).reshape(1, -1)
    return {
        "DP":      (1.0, float(models["c_dp"] * dp_work(volume)), volume <= models["v_max"]),
        "POA":     (float(np.clip(models["score_poa"].predict(v)[0], 0, 1)),
                    float(models["c_poa"] * poa_work(mean_len)), True),
        "SECTION": (float(np.clip(models["score_section"].predict(v)[0], 0, 1)),
                    float(10 ** models["log_time_section"].predict(v)[0]),
                    n_anchors > 0),
    }


def pick(preds, lam=0.01, time_budget=None):
    """Strategy maximizing score - lam * time among allowed ones within the budget."""
    options = {s: (sc, t) for s, (sc, t, ok) in preds.items()
               if ok and (time_budget is None or t <= time_budget)}
    if not options:
        options = {"POA": preds["POA"][:2]}
    return max(options, key=lambda s: options[s][0] - lam * options[s][1])


def choose_strategy(x, y, z, depth=0, lam=0.01, time_budget=None, models=None):
    """Decision for new sequences, using the saved model unless one is passed in."""
    if models is None:
        models = load_models()
    f = extract_features(x, y, z, depth=depth)
    volume = (len(x) + 1) * (len(y) + 1) * (len(z) + 1)
    mean_len = (len(x) + len(y) + len(z)) / 3
    preds = predict_from_features(models, to_vector(f), volume, mean_len, f["n_anchors"])
    return pick(preds, lam, time_budget), preds


def ml_choices(models, d, lam=0.01, time_budget=None):
    """ML strategy choice for every row of d (uses the stored features)."""
    out = []
    for _, row in d.iterrows():
        preds = predict_from_features(models, row[FEATURE_NAMES].values.astype(float),
                                      row["volume"], row["mean_len"], row["n_anchors"])
        out.append(pick(preds, lam, time_budget))
    return np.array(out)


def realized(d, choice):
    """Measured score fraction and time of the chosen strategy for each row."""
    score = {"DP": np.ones(len(d)), "POA": d["score_poa"].values, "SECTION": d["score_section"].values}
    time_ = {"DP": d["time_dp"].values, "POA": d["time_poa"].values, "SECTION": d["time_section"].values}
    sc = np.array([score[c][i] for i, c in enumerate(choice)])
    t = np.array([time_[c][i] for i, c in enumerate(choice)])
    return sc, t


def labelled(df):
    """Rows with a DP optimum and a sectioning result (needed for score fractions)."""
    return df[df["dp_score"].notna() & df["section_score"].notna()].copy()


# ---------------------------------------------------------------------------
# Figures and tables
# ---------------------------------------------------------------------------

def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR}/{name}.png/.pdf")


def save_table(table, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    table.to_csv(os.path.join(FIG_DIR, f"{name}.csv"), index=False)
    with open(os.path.join(FIG_DIR, f"{name}.tex"), "w") as f:
        f.write(table.to_latex(index=False, float_format="%.4f"))
    print(f"Saved {FIG_DIR}/{name}.csv/.tex")


# ---- 4.1 Machine learning model results -----------------------------------

def ml_metrics_table(models, df):
    """Table: test-set accuracy of each learned model vs a mean-value baseline."""
    test = test_split(df, models)
    rest = df[~df["set_id"].isin(set(models["test_set_ids"]))]
    targets = [("POA score", "score_poa", "score_poa", None),
               ("Sectioning score", "score_section", "score_section", None),
               ("Sectioning time (log10 s)", "log_time_section", "time_section",
                lambda t: np.log10(t.where(t > 0)))]
    rows = []
    for name, key, col, tf in targets:
        y_te = test[col].astype(float)
        y_tr = rest[col].astype(float)
        if tf is not None:
            y_te, y_tr = tf(y_te), tf(y_tr)
        ok = np.isfinite(y_te)
        pred = models[key].predict(test.loc[ok, FEATURE_NAMES].values)
        y = y_te[ok].values
        base = np.full_like(y, np.nanmean(y_tr))
        rows.append({"Model": name, "n test": len(y),
                     "MAE": mean_absolute_error(y, pred),
                     "RMSE": float(np.sqrt(np.mean((y - pred) ** 2))),
                     "R2": r2_score(y, pred),
                     "Baseline MAE": mean_absolute_error(y, base)})
    table = pd.DataFrame(rows)
    save_table(table, "table_ml_metrics")
    print("\nTable: ML model test results\n", table.to_string(index=False, float_format="%.4f"))
    return table


def plot_pred_vs_true(models, df):
    """Fig: predicted vs measured values on the test set, one panel per model."""
    test = test_split(df, models)
    panels = [("POA score (fraction of optimum)", "score_poa", "score_poa", None),
              ("Sectioning score (fraction of optimum)", "score_section", "score_section", None),
              ("Sectioning time (log10 s)", "log_time_section", "time_section",
               lambda t: np.log10(t.where(t > 0)))]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (title, key, col, tf) in zip(axes, panels):
        y = test[col].astype(float)
        if tf is not None:
            y = tf(y)
        ok = np.isfinite(y)
        y = y[ok].values
        pred = models[key].predict(test.loc[ok, FEATURE_NAMES].values)
        ax.scatter(y, pred, s=8, alpha=0.35, color=COLORS["OPTIK"], edgecolors="none")
        lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], color="#555555", lw=1, ls="--", label="perfect prediction")
        ax.set_title(title)
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")
        ax.text(0.04, 0.95, f"MAE {mean_absolute_error(y, pred):.4f}\nR² {r2_score(y, pred):.3f}",
                transform=ax.transAxes, va="top", fontsize=9)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, "fig_ml_pred_vs_true")


def policy_table(models, df, lam=0.01):
    """Table: fixed strategies vs OPTIK vs oracle on the test set."""
    d = labelled(test_split(df, models))
    choice = ml_choices(models, d, lam)
    names = ["DP", "POA", "SECTION"]
    score = {"DP": np.ones(len(d)), "POA": d["score_poa"].values, "SECTION": d["score_section"].values}
    time_ = {"DP": d["time_dp"].values, "POA": d["time_poa"].values, "SECTION": d["time_section"].values}
    util = np.stack([score[s] - lam * time_[s] for s in names])
    oracle = np.array(names)[util.argmax(axis=0)]

    policies = {"Always full DP": np.full(len(d), "DP"), "Always POA": np.full(len(d), "POA"),
                "Always sectioning": np.full(len(d), "SECTION"), "OPTIK (ML)": choice, "Oracle": oracle}
    dp_total = time_["DP"].sum()
    rows = []
    for name, c in policies.items():
        sc, t = realized(d, c)
        rows.append({"Policy": name, "Mean score": sc.mean(), "Min score": sc.min(),
                     "Total time (s)": t.sum(), "Speedup vs DP": dp_total / t.sum()})
    table = pd.DataFrame(rows)
    save_table(table, "table_policy")
    print(f"\nTable: policy comparison (lambda = {lam}, {len(d)} test sets)\n",
          table.to_string(index=False, float_format="%.4f"))
    print(f"ML matches oracle in {(choice == oracle).mean():.1%} of sets")
    counts = {s: int((choice == s).sum()) for s in names}
    print(f"ML choices: {counts}")
    return table


def plot_lambda_sweep(models, df, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100)):
    """Fig: accuracy vs runtime trade-off of OPTIK as lambda varies, with fixed strategies."""
    d = labelled(test_split(df, models))
    fig, ax = plt.subplots(figsize=(6, 4))

    pts = []
    for lam in lams:
        sc, t = realized(d, ml_choices(models, d, lam))
        pts.append((t.sum(), sc.mean(), lam))
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], "-o", color=COLORS["OPTIK"], lw=2, ms=6, label=LABELS["OPTIK"])
    for t, s, lam in pts:
        ax.annotate(f"λ={lam:g}", (t, s), textcoords="offset points", xytext=(5, -10), fontsize=7,
                    color="#555555")

    for m in ["DP", "POA", "SECTION"]:
        sc, t = realized(d, np.full(len(d), m))
        ax.scatter(t.sum(), sc.mean(), s=70, color=COLORS[m], zorder=3, label=f"Always {LABELS[m]}")

    ax.set_xscale("log")
    ax.set_xlabel("Total runtime on test set (s, log scale)")
    ax.set_ylabel("Mean score (fraction of optimum)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_lambda_sweep")


# ---- 4.2 Synthetic sequence results ----------------------------------------

def size_bin(mean_len):
    """
    Size class by upper bound:
      <= 20 -> 20,  21-100 -> 100,  101-500 -> 500,  501-1000 -> 1000,  > 1000 -> excluded
    """
    if mean_len > 1000:
        return np.nan
    if mean_len > 500:
        return 1000
    if mean_len > 100:
        return 500
    if mean_len > 20:
        return 100
    return 20


def plot_size_bars(models, df, lam=0.01, use_test_only=True, y_min=90):
    """
    Fig: grouped bars by sequence size, one bar per method.
      bins: <=20, 21-100, 101-500, 501-1000 bp; sets > 1000 bp are excluded
      (a) mean runtime (log scale)
      (b) mean score as % of the optimum, y-axis from y_min to 100
    """
    d = labelled(test_split(df, models) if use_test_only else df)
    d["size"] = d["mean_len"].apply(size_bin)
    n_excluded = int(d["size"].isna().sum())
    d = d[d["size"].notna()].copy()
    d["size"] = d["size"].astype(int)
    if n_excluded:
        print(f"Excluded {n_excluded} sets longer than 1000 bp from the size plot")

    d["OPTIK_choice"] = ml_choices(models, d, lam)

    per_method = {}
    for m in METHODS:
        choice = d["OPTIK_choice"].values if m == "OPTIK" else np.full(len(d), m)
        per_method[m] = realized(d, choice)

    sizes = [s for s in SIZES if (d["size"] == s).any()]

    rows = []
    for s in sizes:
        mask = (d["size"] == s).values
        n = int(mask.sum())
        for m in METHODS:
            sc, t = per_method[m]
            pct = 100 * sc[mask]
            rows.append({"size": s, "method": m, "n": n,
                         "time_mean": t[mask].mean(),
                         "time_sem": t[mask].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0,
                         "score_pct_mean": pct.mean(),
                         "score_pct_sem": pct.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0})
    summary = pd.DataFrame(rows)
    save_table(summary, "table_size_summary")

    below = summary[summary["score_pct_mean"] < y_min]
    if len(below):
        print(f"Warning: {len(below)} bars are below {y_min}% and will be cut off:\n",
              below[["size", "method", "score_pct_mean"]].to_string(index=False))

    x = np.arange(len(sizes))
    w = 0.8 / len(METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for k, m in enumerate(METHODS):
        sub = summary[summary["method"] == m].set_index("size").loc[sizes]
        pos = x - 0.4 + w / 2 + k * w
        axes[0].bar(pos, sub["time_mean"], w * 0.92, yerr=sub["time_sem"], color=COLORS[m],
                    label=LABELS[m], capsize=2, error_kw={"lw": 0.8})
        heights = np.maximum(sub["score_pct_mean"].values - y_min, 0)
        axes[1].bar(pos, heights, w * 0.92, bottom=y_min, yerr=sub["score_pct_sem"].values,
                    color=COLORS[m], label=LABELS[m], capsize=2, error_kw={"lw": 0.8})

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([BIN_LABELS[s] for s in sizes])
        ax.set_xlabel("Sequence length (bp)")
        ax.grid(axis="x", visible=False)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean runtime per set (s, log scale)")
    axes[0].set_title("(a) Runtime")
    axes[1].set_ylim(y_min, 100.5)
    axes[1].set_ylabel("Score (% of optimum)")
    axes[1].set_title(f"(b) Score (y-axis from {y_min}%)")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    n_text = ",  ".join(f"{BIN_LABELS[s]} bp: n={int((d['size'] == s).sum())}" for s in sizes)
    fig.text(0.5, -0.02, n_text, ha="center", fontsize=8, color="#555555")
    fig.tight_layout()
    save(fig, "fig_size_bars")
    print("\nSize summary\n", summary.to_string(index=False, float_format="%.4g"))
    return summary


def make_all_figures(models, df, lam=0.01):
    ml_metrics_table(models, df)            # 4.1 table: model accuracy
    plot_pred_vs_true(models, df)           # 4.1 figure: predicted vs measured
    policy_table(models, df, lam)           # 4.1 table: policies vs oracle
    plot_lambda_sweep(models, df)           # 4.1 figure: score vs runtime trade-off
    plot_size_bars(models, df, lam)         # 4.2 figure: methods by sequence size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    RETRAIN = False                     # set True to train and overwrite the saved model
    LAMBDA = 0.01

    df = load_results(RESULTS_FILE)

    if RETRAIN or not os.path.exists(MODEL_FILE):
        train(df)

    models = load_models()
    make_all_figures(models, df, lam=LAMBDA)

    # Predict for new sequences with the saved model
    choice, preds = choose_strategy("CTATAGAGTA", "ATTATGAGTC", "CTATTGAGTA")
    print("\nExample choice:", choice)
    for s, (sc, t, ok) in preds.items():
        print(f"  {s:>7}: score {sc:.3f}, time {t:.2e} s, allowed {ok}")