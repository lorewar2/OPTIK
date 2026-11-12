import math
import os
import time
import numpy as np
import pandas as pd
import spoa
from multiprocessing import Pool

from full_dp import dp3, MATCH, MISMATCH, SKIP
from n_lcsk import lcskpp_3way, path_to_anchors, unanchored_regions, max_k
from feature_extractor import extract_features, read_fasta

V_MAX = 1e10          # largest DP volume (cells) per run; 1e10 cells = 50 GB
K_MIN = 4            # smallest k allowed for anchors
MIN_ANCHORS = 1      # anchors required at the maximal k
N_WORKERS = 10       # parallel processes; keep N_WORKERS * 5 * V_MAX bytes below your RAM


# ---------------------------------------------------------------------------
# Scoring (same rule as dp3)
# ---------------------------------------------------------------------------

def step_score(bases, n_skip):
    s = 0
    if len(bases) >= 2:
        s = MATCH * (len(bases) - 1) if len(set(bases)) == 1 else MISMATCH
    return s + SKIP * n_skip


def score_msa(msa):
    total = 0
    for col in zip(*msa):
        bases = [b for b in col if b != "-"]
        if bases:
            total += step_score(bases, len(col) - len(bases))
    return total


def volume(seqs):
    return math.prod(len(s) + 1 for s in seqs)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def run_dp(seqs):
    if volume(seqs) > V_MAX:
        return None, float("nan")
    t0 = time.perf_counter()
    _, msa, _ = dp3(*seqs)
    return msa, time.perf_counter() - t0


def run_poa(seqs):
    t0 = time.perf_counter()
    nonempty = [n for n, s in enumerate(seqs) if s]
    if len(nonempty) == 0:
        msa = ["" for _ in seqs]
    elif len(nonempty) == 1:
        s = seqs[nonempty[0]]
        msa = [s if n == nonempty[0] else "-" * len(s) for n in range(len(seqs))]
    else:
        _, sub = spoa.poa([seqs[n] for n in nonempty])
        width = len(sub[0])
        rows = iter(sub)
        msa = [next(rows) if n in nonempty else "-" * width for n in range(len(seqs))]
    return msa, time.perf_counter() - t0

def run_sectioning(x, y, z):
    t0 = time.perf_counter()
    k = max_k(x, y, z, k_min=K_MIN, min_anchors=MIN_ANCHORS)
    if k is None:
        return None
    _, path = lcskpp_3way(x, y, z, k)
    anchors = path_to_anchors(path, k)
    lcsk_time = time.perf_counter() - t0

    starts = [(0, 0, 0)] + [(i + L, j + L, l + L) for i, j, l, L in anchors]
    ends = [(i, j, l) for i, j, l, L in anchors] + [(len(x), len(y), len(z))]

    out = {"k_used": k, "n_anchors": len(anchors), "lcsk_time": lcsk_time,
           "anchor_bases": sum(a[3] for a in anchors)}

    for mode in ["section", "section_poa"]:
        parts, times, n_dp, n_regions, max_region_vol = [], [], 0, 0, 0
        region_sizes = []
        for r, (s, e) in enumerate(zip(starts, ends)):
            region = [x[s[0]:e[0]], y[s[1]:e[1]], z[s[2]:e[2]]]
            if any(region):
                n_regions += 1
                region_sizes.append("/".join(str(len(q)) for q in region))
                max_region_vol = max(max_region_vol, volume(region))
                msa = None
                if mode == "section":
                    msa, t = run_dp(region)
                    n_dp += msa is not None
                if msa is None:
                    msa, t = run_poa(region)
                parts.append(msa)
                times.append(t)
            if r < len(anchors):
                i, j, l, L = anchors[r]
                parts.append([x[i:i + L]] * 3)

        msa = ["".join(p[row] for p in parts) for row in range(3)]
        out[f"{mode}_score"] = score_msa(msa)
        out[f"{mode}_time"] = lcsk_time + (max(times) if times else 0.0)   # regions in parallel
        out[f"{mode}_time_serial"] = lcsk_time + sum(times)
        if mode == "section":
            out["n_regions"] = n_regions
            out["section_regions_dp"] = n_dp
            out["max_region_volume"] = max_region_vol
            out["region_sizes"] = ";".join(region_sizes)       # e.g. "3/2/3;40/38/41"
            out["anchor_lengths"] = ";".join(str(a[3]) for a in anchors)

    return out

# ---------------------------------------------------------------------------
# One set (runs inside a worker process)
# ---------------------------------------------------------------------------

def process_set(item):
    set_id, (x, y, z) = item
    seqs = [x, y, z]
    try:
        row = {"set_id": set_id, "len_x": len(x), "len_y": len(y), "len_z": len(z),
               "volume": volume(seqs), "mean_len": float(np.mean([len(s) for s in seqs]))}

        row.update(extract_features(x, y, z, depth=0))

        msa, t = run_dp(seqs)
        row["dp_score"] = score_msa(msa) if msa is not None else float("nan")
        row["dp_time"] = t

        msa, t = run_poa(seqs)
        row["poa_score"] = score_msa(msa)
        row["poa_time"] = t

        sec = run_sectioning(x, y, z)
        if sec:
            row.update(sec)

        opt = row["dp_score"]
        for m in ["poa", "section", "section_poa"]:
            val = row.get(f"{m}_score", float("nan"))
            row[f"{m}_frac"] = val / opt if not math.isnan(opt) and opt != 0 else float("nan")

        row["score_poa"] = row["poa_frac"]
        row["score_section"] = row["section_frac"]
        row["time_dp"] = row["dp_time"]
        row["time_poa"] = row["poa_time"]
        row["time_section"] = row.get("section_time", float("nan"))
        row["error"] = ""
    except Exception as e:                               # keep going if one set fails
        row = {"set_id": set_id, "error": repr(e)}
    return row


def init_worker():
    """Each worker compiles Numba once (cache=True makes later workers fast)."""
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    dp3("ACGT", "ACG", "AGT")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_data(fasta_file="sequences.fasta", out_file="results.csv", n_workers=N_WORKERS):
    sets = {}
    for name, seq in read_fasta(fasta_file):
        set_id, tag = name.split("-", 1)
        if tag != "original":
            sets.setdefault(set_id, []).append(seq)
    items = [(sid, tuple(s)) for sid, s in sets.items() if len(s) == 3]

    dp3("ACGT", "ACG", "AGT")

    done, columns = 0, None
    t_start = time.perf_counter()
    with Pool(n_workers, initializer=init_worker) as pool:
        for row in pool.imap_unordered(process_set, items, chunksize=1):
            df_row = pd.DataFrame([row])
            if columns is None:
                columns = list(df_row.columns)
                df_row.to_csv(out_file, index=False)
            else:
                df_row.reindex(columns=columns).to_csv(out_file, mode="a", header=False, index=False)
            done += 1
            print_row(row, done, len(items), time.perf_counter() - t_start)

    df = pd.read_csv(out_file)
    df["set_id_num"] = pd.to_numeric(df["set_id"], errors="coerce")
    df = df.sort_values("set_id_num").drop(columns="set_id_num")
    df.to_csv(out_file, index=False)
    print(f"\nWrote {len(df)} rows to {out_file}")
    return df


def print_row(row, done, total, elapsed, max_regions_shown=10):
    """Readable summary of one set: scores, times and sections."""
    head = f"[{done}/{total}  {elapsed:.0f}s]  set {row['set_id']}"
    if row.get("error"):
        print(f"{head}: ERROR {row['error']}", flush=True)
        return

    lines = [f"{head}  len {row['len_x']}/{row['len_y']}/{row['len_z']}  volume {row['volume']:.2e}",
             f"    DP    score {row['dp_score']:>8}   time {row['dp_time']:.4f} s",
             f"    POA   score {row['poa_score']:>8}   time {row['poa_time']:.4f} s"]

    if "section_score" in row:
        sizes = row["region_sizes"].split(";") if row["region_sizes"] else []
        shown = ", ".join(sizes[:max_regions_shown]) + (" ..." if len(sizes) > max_regions_shown else "")
        lines += [
            f"    SEC   score {row['section_score']:>8}   time {row['section_time']:.4f} s"
            f"  (serial {row['section_time_serial']:.4f} s, LCSk++ {row['lcsk_time']:.4f} s)",
            f"          k={row['k_used']}  anchors={row['n_anchors']} (lengths {row['anchor_lengths']})",
            f"          sections={row['n_regions']}  DP on {row['section_regions_dp']}  "
            f"largest volume {row['max_region_volume']:.2e}",
            f"          section lengths X/Y/Z: {shown}",
        ]
    else:
        lines.append("    SEC   none (no anchors at k >= K_MIN)")

    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    make_data("./data/sequences.fasta", "./data/results.csv", n_workers=N_WORKERS)