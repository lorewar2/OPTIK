from collections import defaultdict

# ---------------------------------------------------------------------------
# k-mer finding
# ---------------------------------------------------------------------------

def kmer_index(seq, k):
    """Map each k-mer to the list of positions where it starts (0-based)."""
    index = defaultdict(list)
    for i in range(len(seq) - k + 1):
        index[seq[i:i + k]].append(i)
    return index


def three_way_matches(x, y, z, k):
    """All (i, j, l) where the k-mer starting at i in x, j in y, l in z is identical."""
    ix, iy, iz = kmer_index(x, k), kmer_index(y, k), kmer_index(z, k)
    matches = []
    for kmer, xs in ix.items():
        if kmer in iy and kmer in iz:
            for i in xs:
                for j in iy[kmer]:
                    for l in iz[kmer]:
                        matches.append((i, j, l))
    return sorted(matches)


def pairwise_matches(a, b, k):
    """All (p, q) where the k-mer starting at p in a equals the one at q in b."""
    ia, ib = kmer_index(a, k), kmer_index(b, k)
    return sorted((p, q) for kmer, ps in ia.items() if kmer in ib
                  for p in ps for q in ib[kmer])


def pair_only_matches(x, y, z, k, m3=None):
    """
    Pairwise matches not contained in any 3-way match.
    Returns {"XY": [(i, j)], "XZ": [(i, l)], "YZ": [(j, l)]}.
    """
    if m3 is None:
        m3 = three_way_matches(x, y, z, k)
    used_xy = {(i, j) for i, j, l in m3}
    used_xz = {(i, l) for i, j, l in m3}
    used_yz = {(j, l) for i, j, l in m3}
    return {
        "XY": [m for m in pairwise_matches(x, y, k) if m not in used_xy],
        "XZ": [m for m in pairwise_matches(x, z, k) if m not in used_xz],
        "YZ": [m for m in pairwise_matches(y, z, k) if m not in used_yz],
    }


# ---------------------------------------------------------------------------
# 3-way LCSk++ chaining
# ---------------------------------------------------------------------------

def lcskpp_3way(x, y, z, k, matches=None):
    """
    3-way LCSk++. Returns (score, path), where path is the list of
    matches (i, j, l) on the best chain, in order.

    score(P) = max( k,
                    score(Q) + k   if Q ends before P starts in all three sequences,
                    score(Q) + 1   if P continues Q, i.e. P = Q + (1, 1, 1) )
    """
    if matches is None:
        matches = three_way_matches(x, y, z, k)
    if not matches:
        return 0, []

    pos = {m: n for n, m in enumerate(matches)}      # match -> index, for continuation lookup
    score = [k] * len(matches)
    prev = [None] * len(matches)

    for p, (pi, pj, pl) in enumerate(matches):       # sorted by i, so predecessors come first
        # continues: Q = P - (1, 1, 1)
        q = pos.get((pi - 1, pj - 1, pl - 1))
        if q is not None and score[q] + 1 > score[p]:
            score[p], prev[p] = score[q] + 1, q
        # precedes: Q ends at or before P starts in all three sequences
        for q, (qi, qj, ql) in enumerate(matches[:p]):
            if qi + k <= pi and qj + k <= pj and ql + k <= pl and score[q] + k > score[p]:
                score[p], prev[p] = score[q] + k, q

    best = max(range(len(matches)), key=score.__getitem__)
    path = []
    while best is not None:
        path.append(matches[best])
        best = prev[best]
    path.reverse()
    return max(score), path


def path_to_anchors(path, k):
    """Merge continuing matches on the path into anchors: (i, j, l, length)."""
    anchors = []
    for i, j, l in path:
        if anchors:
            ai, aj, al, length = anchors[-1]
            if (i, j, l) == (ai + length - k + 1, aj + length - k + 1, al + length - k + 1):
                anchors[-1] = (ai, aj, al, length + 1)   # extends the previous run by one
                continue
        anchors.append((i, j, l, k))
    return anchors


def unanchored_regions(x, y, z, anchors):
    """
    Regions between anchors (plus before the first and after the last).
    Each region is ((x_start, x_end), (y_start, y_end), (z_start, z_end)), end exclusive.
    """
    regions = []
    prev_end = (0, 0, 0)
    for i, j, l, length in anchors:
        regions.append(((prev_end[0], i), (prev_end[1], j), (prev_end[2], l)))
        prev_end = (i + length, j + length, l + length)
    regions.append(((prev_end[0], len(x)), (prev_end[1], len(y)), (prev_end[2], len(z))))
    return [r for r in regions if any(e > s for s, e in r)]   # drop fully empty regions


# ---------------------------------------------------------------------------
# Maximal k
# ---------------------------------------------------------------------------

def max_k(x, y, z, k_min=2, min_anchors=1):
    """
    Largest k (>= k_min) whose LCSk++ path gives at least min_anchors anchors.
    Binary search for the largest k with any 3-way match, then step down
    until the anchor requirement is met. Returns None if no k works.
    """
    lo, hi = k_min, min(len(x), len(y), len(z))
    if lo > hi or not three_way_matches(x, y, z, lo):
        return None
    while lo < hi:                                     # existence of a match shrinks with k
        mid = (lo + hi + 1) // 2
        if three_way_matches(x, y, z, mid):
            lo = mid
        else:
            hi = mid - 1
    for k in range(lo, k_min - 1, -1):
        _, path = lcskpp_3way(x, y, z, k)
        if len(path_to_anchors(path, k)) >= min_anchors:
            return k
    return None


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

x, y, z = "CTATAGAGTA", "ATTATGAGTC", "CTATTGAGTA"
k = 2

m3 = three_way_matches(x, y, z, k)
print("3-way matches:", m3)
print("Pair-only matches:", pair_only_matches(x, y, z, k, m3))

score, path = lcskpp_3way(x, y, z, k, m3)
anchors = path_to_anchors(path, k)
print("LCSk++ score:", score)
print("Path:", path)
print("Anchors (i, j, l, length):", anchors)
for (xs, xe), (ys, ye), (zs, ze) in unanchored_regions(x, y, z, anchors):
    print("Region:", x[xs:xe] or "-", y[ys:ye] or "-", z[zs:ze] or "-")

print("Maximal k (>= 2 anchors):", max_k(x, y, z, min_anchors=2))