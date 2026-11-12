import math
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

from feature_extractor import FEATURE_NAMES, extract_features, to_vector

V_MAX = 1e8          # largest DP volume allowed (cells)
SEED = 42

# ---------------------------------------------------------------------------
# Expected training table: one row per region, with columns
#   set_id                      - which generated set the region came from (for splitting)
#   <FEATURE_NAMES>             - from extract_features
#   score_poa, score_section    - score / optimal score (0..1)
#   time_dp, time_poa, time_section - measured runtimes in seconds
#   volume, mean_len            - DP cells and mean sequence length (for the time formulas)
# ---------------------------------------------------------------------------

def split_by_set(df, seed=SEED):
    """70/15/15 split by set_id, so regions from one set stay in one split."""
    groups = df["set_id"]
    gss = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, rest_idx = next(gss.split(df, groups=groups))
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
    ok = (work > 0) & (seconds > 0)
    return float(np.exp(np.mean(np.log(seconds[ok]) - np.log(work[ok]))))


def dp_work(volume):
    return 7 * volume                      # 7 moves per cell for N = 3

def poa_work(mean_len, n=3):
    return n * mean_len ** 2               # each sequence aligned to a graph of ~L nodes


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_model():
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
        early_stopping=True, validation_fraction=0.1, random_state=SEED)


def train(df):
    train_df, val_df, test_df = split_by_set(df)
    X_tr, X_va, X_te = (d[FEATURE_NAMES].values for d in (train_df, val_df, test_df))

    models = {}

    # 1-2: scores (fraction of optimum)
    for target in ["score_poa", "score_section"]:
        m = make_model().fit(X_tr, train_df[target])
        models[target] = m
        print(f"{target:>15}  val MAE = {mean_absolute_error(val_df[target], m.predict(X_va)):.4f}"
              f"  test MAE = {mean_absolute_error(test_df[target], m.predict(X_te)):.4f}")

    # 3: sectioning time, predicted in log10 seconds
    y_tr = np.log10(train_df["time_section"])
    m = make_model().fit(X_tr, y_tr)
    models["log_time_section"] = m
    for name, d, X in [("val", val_df, X_va), ("test", test_df, X_te)]:
        err = mean_absolute_error(np.log10(d["time_section"]), m.predict(X))
        print(f"time_section  {name} MAE = {err:.3f} log10 s  (x{10 ** err:.2f})")

    # 4-5: time constants for DP and POA
    models["c_dp"] = fit_time_constant(dp_work(train_df["volume"]), train_df["time_dp"])
    models["c_poa"] = fit_time_constant(poa_work(train_df["mean_len"]), train_df["time_poa"])
    print(f"c_dp = {models['c_dp']:.3e} s/cell-move,  c_poa = {models['c_poa']:.3e} s/unit")

    joblib.dump(models, "optik_models.joblib")
    return models, test_df


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def predict_all(models, x, y, z, depth=0):
    """Predicted (score, time) for each strategy on one input."""
    f = extract_features(x, y, z, depth=depth)
    v = to_vector(f).reshape(1, -1)
    volume = (len(x) + 1) * (len(y) + 1) * (len(z) + 1)
    mean_len = (len(x) + len(y) + len(z)) / 3

    return {
        "DP":      (1.0, models["c_dp"] * dp_work(volume), volume <= V_MAX),
        "POA":     (float(np.clip(models["score_poa"].predict(v)[0], 0, 1)),
                    models["c_poa"] * poa_work(mean_len), True),
        "SECTION": (float(np.clip(models["score_section"].predict(v)[0], 0, 1)),
                    float(10 ** models["log_time_section"].predict(v)[0]),
                    f["n_anchors"] > 0),
    }


def choose_strategy(models, x, y, z, depth=0, lam=0.01, time_budget=None):
    """
    Pick the strategy maximizing  score - lam * time,
    among those that are allowed and fit the time budget.
    lam = score fraction you would give up per second.
    """
    preds = predict_all(models, x, y, z, depth)
    options = {s: (sc, t) for s, (sc, t, ok) in preds.items()
               if ok and (time_budget is None or t <= time_budget)}
    if not options:                       # nothing fits: fall back to the fastest
        options = {"POA": preds["POA"][:2]}
    best = max(options, key=lambda s: options[s][0] - lam * options[s][1])
    return best, preds


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv("training_table.csv")
    models, test_df = train(df)

    # models = joblib.load("optik_models.joblib")
    choice, preds = choose_strategy(models, "CTATAGAGTA", "ATTATGAGTC", "CTATTGAGTA")
    print("Choice:", choice)
    for s, (sc, t, ok) in preds.items():
        print(f"  {s:>7}: score {sc:.3f}, time {t:.2e} s, allowed {ok}")