import spoa   # pip install pyspoa

MATCH, MISMATCH, SKIP = 2, -2, -1

def step_score(bases, n_skip):
    """Same scoring as the 3D DP: bases = bases of the sequences that advance."""
    s = 0
    if len(bases) >= 2:
        s = MATCH * (len(bases) - 1) if len(set(bases)) == 1 else MISMATCH
    return s + SKIP * n_skip

def score_msa(msa):
    """Score an MSA (list of equal-length rows with '-') as a path through the DP cube."""
    total = 0
    for col in zip(*msa):
        bases = [b for b in col if b != "-"]
        if bases:                                   # skip all-gap columns
            total += step_score(bases, len(col) - len(bases))
    return total
