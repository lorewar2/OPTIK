import math
import numpy as np
import pandas as pd

from n_lcsk import kmer_index, lcskpp_3way, path_to_anchors, unanchored_regions, max_k

K_LIST = [4, 6, 8, 12, 16]     # k values for the k-mer and LCSk++ features
MAX_MATCHES = 200_000          # skip LCSk++ at a k if there are too many matches (keeps features cheap)


# ---------------------------------------------------------------------------
# k-mer features
# ---------------------------------------------------------------------------

def kmer_features(x, y, z, k):
    """
    Shared k-mer content at one k:
      shared3  - fraction of distinct k-mers (of the shortest sequence) found in all three
      pairXY/XZ/YZ - fraction of distinct k-mers shared by that pair only (absent from the third)
      n3       - number of 3-way match tuples (position combinations), log10
      lcsk     - LCSk++ score / shortest length (-1 if skipped because of too many matches)
    """
    ix, iy, iz = kmer_index(x, k), kmer_index(y, k), kmer_index(z, k)
    sx, sy, sz = set(ix), set(iy), set(iz)
    min_distinct = max(1, min(len(sx), len(sy), len(sz)))

    shared3 = sx & sy & sz
    pair_xy = (sx & sy) - sz
    pair_xz = (sx & sz) - sy
    pair_yz = (sy & sz) - sx

    # number of (i, j, l) match tuples, counted without listing them
    n3 = sum(len(ix[m]) * len(iy[m]) * len(iz[m]) for m in shared3)

    if 0 < n3 <= MAX_MATCHES:
        matches = sorted((i, j, l) for m in shared3
                         for i in ix[m] for j in iy[m] for l in iz[m])
        score, _ = lcskpp_3way(x, y, z, k, matches)
        lcsk = score / max(1, min(len(x), len(y), len(z)))
    elif n3 == 0:
        lcsk = 0.0
    else:
        lcsk = -1.0                                    # too many matches: skipped

    return {
        f"shared3_k{k}": len(shared3) / min_distinct,
        f"pairXY_k{k}": len(pair_xy) / min_distinct,
        f"pairXZ_k{k}": len(pair_xz) / min_distinct,
        f"pairYZ_k{k}": len(pair_yz) / min_distinct,
        f"log_n3_k{k}": math.log10(1 + n3),
        f"lcsk_k{k}": lcsk,
    }


# ---------------------------------------------------------------------------
# Sequence content features
# ---------------------------------------------------------------------------

def homopolymer_fraction(seq, min_run=4):
    """Fraction of bases inside homopolymer runs of length >= min_run."""
    if not seq:
        return 0.0
    covered, run = 0, 1
    for a, b in zip(seq, seq[1:]):
        if a == b:
            run += 1
        else:
            if run >= min_run:
                covered += run
            run = 1
    if run >= min_run:
        covered += run
    return covered / len(seq)


def repeat_fraction(seq, k=8):
    """Fraction of k-mer positions whose k-mer occurs more than once in the sequence."""
    n = len(seq) - k + 1
    if n <= 0:
        return 0.0
    idx = kmer_index(seq, k)
    return sum(len(p) for p in idx.values() if len(p) > 1) / n


def gc_content(seq):
    return (seq.count("G") + seq.count("C")) / max(1, len(seq))


# ---------------------------------------------------------------------------
# Full feature vector for one region
# ---------------------------------------------------------------------------

def extract_features(x, y, z, depth=0, k_min=4, min_anchors=2):
    """Feature dictionary for one input (three sequences or one unanchored region)."""
    lx, ly, lz = len(x), len(y), len(z)
    lengths = [lx, ly, lz]
    volume = (lx + 1) * (ly + 1) * (lz + 1)
    min_len = max(1, min(lengths))

    f = {}

    # --- size ---
    f["log_len_x"] = math.log10(1 + lx)
    f["log_len_y"] = math.log10(1 + ly)
    f["log_len_z"] = math.log10(1 + lz)
    f["log_len_min"] = math.log10(1 + min(lengths))
    f["log_len_max"] = math.log10(1 + max(lengths))
    f["len_imbalance"] = (max(lengths) - min(lengths)) / max(1, max(lengths))
    f["log_volume"] = math.log10(volume)
    f["n_empty"] = sum(l == 0 for l in lengths)      # 1 or 2 means a degenerate region

    # --- estimated runtime (relative units) ---
    f["log_est_dp"] = math.log10(7 * volume)                       # 7 moves per cell
    f["log_est_poa"] = math.log10(1 + 3 * (np.mean(lengths) ** 2))  # ~ N * L * graph size

    # --- maximal-k anchors ---
    kmax = max_k(x, y, z, k_min=k_min, min_anchors=min_anchors) if min(lengths) >= k_min else None
    if kmax is None:
        f.update({"max_k": 0, "max_k_frac": 0.0, "n_anchors": 0, "anchor_coverage": 0.0,
                  "largest_region_frac": 1.0, "log_largest_region_vol": f["log_volume"]})
    else:
        _, path = lcskpp_3way(x, y, z, kmax)
        anchors = path_to_anchors(path, kmax)
        regions = unanchored_regions(x, y, z, anchors)
        region_vols = [(xe - xs + 1) * (ye - ys + 1) * (ze - zs + 1)
                       for (xs, xe), (ys, ye), (zs, ze) in regions] or [1]
        f["max_k"] = kmax
        f["max_k_frac"] = kmax / min_len
        f["n_anchors"] = len(anchors)
        f["anchor_coverage"] = sum(a[3] for a in anchors) / min_len
        f["largest_region_frac"] = max(region_vols) / volume
        f["log_largest_region_vol"] = math.log10(max(region_vols))

    # --- shared k-mers and LCSk++ at several k ---
    for k in K_LIST:
        if min(lengths) >= k:
            f.update(kmer_features(x, y, z, k))
        else:
            f.update({f"shared3_k{k}": 0.0, f"pairXY_k{k}": 0.0, f"pairXZ_k{k}": 0.0,
                      f"pairYZ_k{k}": 0.0, f"log_n3_k{k}": 0.0, f"lcsk_k{k}": 0.0})

    # --- sequence content (mean over the three) ---
    seqs = [s for s in (x, y, z) if s]
    f["homopolymer_frac"] = float(np.mean([homopolymer_fraction(s) for s in seqs])) if seqs else 0.0
    f["repeat_frac"] = float(np.mean([repeat_fraction(s) for s in seqs])) if seqs else 0.0
    f["gc"] = float(np.mean([gc_content(s) for s in seqs])) if seqs else 0.0

    # --- recursion ---
    f["depth"] = depth

    return f


# Fixed column order for the model
FEATURE_NAMES = list(extract_features("ACGTACGTACGTACGT", "ACGTACGTACGTACGT", "ACGTACGTACGTACGT").keys())


def to_vector(features):
    """Feature dict -> numpy vector in FEATURE_NAMES order."""
    return np.array([features[n] for n in FEATURE_NAMES], dtype=np.float32)


# ---------------------------------------------------------------------------
# Build a feature table from the generator's FASTA
# ---------------------------------------------------------------------------

def read_fasta(filename):
    """Returns a list of (name, sequence); the name excludes the 'length=' part."""
    records, name, seq = [], None, []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(seq)))
                name, seq = line[1:].split()[0], []
            else:
                seq.append(line)
    if name is not None:
        records.append((name, "".join(seq)))
    return records


def build_feature_table(fasta_file):
    """One row per set: features of its three copies (the original is skipped)."""
    sets = {}
    for name, seq in read_fasta(fasta_file):
        set_id, tag = name.split("-", 1)
        if tag != "original":
            sets.setdefault(set_id, []).append(seq)

    rows = []
    for set_id, seqs in sets.items():
        if len(seqs) != 3:
            continue
        f = extract_features(*seqs)
        f["set_id"] = set_id
        rows.append(f)

    return pd.DataFrame(rows)[["set_id"] + FEATURE_NAMES]