import itertools
import numpy as np

MATCH, MISMATCH, SKIP = 2, -2, -1

def step_score(bases, n_skip):
    """Score of one step: bases = bases of the sequences that advance."""
    s = 0
    if len(bases) >= 2:
        s = MATCH * (len(bases) - 1) if len(set(bases)) == 1 else MISMATCH
    return s + SKIP * n_skip


def nd_dp(seqs):
    """
    Exact N-dimensional DP over len(seqs) sequences.
    Returns (optimal score, MSA rows with '-', path of cells from origin to end).
    """
    n = len(seqs)
    shape = tuple(len(s) + 1 for s in seqs)
    moves = [m for m in itertools.product((0, 1), repeat=n) if any(m)]   # 2^N - 1 moves

    D = np.full(shape, -np.inf)
    ptr = np.full(shape, -1, dtype=np.int16)
    D[(0,) * n] = 0

    # itertools.product visits cells in lexicographic order,
    # so every predecessor is filled before the cell that needs it
    for cell in itertools.product(*(range(s) for s in shape)):
        if not any(cell):
            continue
        best, best_m = -np.inf, -1
        for mi, m in enumerate(moves):
            prev = tuple(c - d for c, d in zip(cell, m))
            if min(prev) < 0:
                continue
            bases = [seqs[k][cell[k] - 1] for k in range(n) if m[k]]
            val = D[prev] + step_score(bases, n - sum(m))
            if val > best:
                best, best_m = val, mi
        D[cell] = best
        ptr[cell] = best_m

    # Traceback from the final cell
    end = tuple(s - 1 for s in shape)
    cell, cols, path = end, [], [end]
    while any(cell):
        m = moves[ptr[cell]]
        cols.append([seqs[k][cell[k] - 1] if m[k] else "-" for k in range(n)])
        cell = tuple(c - d for c, d in zip(cell, m))
        path.append(cell)
    cols.reverse()
    path.reverse()
    msa = ["".join(col[k] for col in cols) for k in range(n)]

    return D[end], msa, path


def consensus(msa):
    """Majority base per column; columns where most rows are '-' are dropped."""
    out = []
    for col in zip(*msa):
        bases = [b for b in col if b != "-"]
        if len(bases) > len(col) / 2:
            out.append(max(set(bases), key=bases.count))
    return "".join(out)


seqs = ["CTATAGAGTA", "ATTATGAGTC", "CTATTGAGTA"]
score, msa, path = nd_dp(seqs)
print("Optimal score:", score)
for row in msa:
    print(row)
print("Consensus:", consensus(msa))