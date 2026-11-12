import numpy as np
from numba import njit

MATCH, MISMATCH, SKIP = 2, -2, -2
NEG = -(2 ** 30)                                   # "minus infinity" for int32

# The 7 moves: (a, b, d) = which of X, Y, Z advance
MA = np.array([1, 0, 0, 1, 1, 0, 1], dtype=np.int64)
MB = np.array([0, 1, 0, 1, 0, 1, 1], dtype=np.int64)
MD = np.array([0, 0, 1, 0, 1, 1, 1], dtype=np.int64)


@njit(cache=True)
def _step(bx, by, bz, a, b, d):
    """Step score: same rule as step_score() in the Python DP."""
    n = a + b + d
    s = 0
    if n >= 2:
        if n == 3:
            same = (bx == by) and (by == bz)
        elif a and b:
            same = bx == by
        elif a and d:
            same = bx == bz
        else:
            same = by == bz
        s = MATCH * (n - 1) if same else MISMATCH
    return s + SKIP * (3 - n)


@njit(cache=True)
def _fill(x, y, z):
    nx, ny, nz = len(x), len(y), len(z)
    D = np.full((nx + 1, ny + 1, nz + 1), NEG, dtype=np.int32)
    ptr = np.full((nx + 1, ny + 1, nz + 1), -1, dtype=np.int8)
    D[0, 0, 0] = 0

    for i in range(nx + 1):
        for j in range(ny + 1):
            for l in range(nz + 1):
                if i == 0 and j == 0 and l == 0:
                    continue
                best = NEG
                best_m = -1
                for m in range(7):
                    a, b, d = MA[m], MB[m], MD[m]
                    pi, pj, pl = i - a, j - b, l - d
                    if pi < 0 or pj < 0 or pl < 0:
                        continue
                    prev = D[pi, pj, pl]
                    if prev == NEG:
                        continue
                    bx = x[i - 1] if a else 0
                    by = y[j - 1] if b else 0
                    bz = z[l - 1] if d else 0
                    val = prev + _step(bx, by, bz, a, b, d)
                    if val > best:
                        best = val
                        best_m = m
                D[i, j, l] = best
                ptr[i, j, l] = best_m
    return D, ptr


@njit(cache=True)
def _traceback(ptr, nx, ny, nz):
    """Moves from the end cell back to the origin, in forward order."""
    moves = np.empty(nx + ny + nz, dtype=np.int8)
    n = 0
    i, j, l = nx, ny, nz
    while i > 0 or j > 0 or l > 0:
        m = ptr[i, j, l]
        moves[n] = m
        n += 1
        i -= MA[m]
        j -= MB[m]
        l -= MD[m]
    return moves[:n][::-1]


def dp3(x, y, z):
    """
    Exact 3D DP (Numba). Returns (optimal score, MSA rows with '-', path of cells).
    Memory: 5 bytes per cell (int32 scores + int8 pointers).
    """
    ex = np.frombuffer(x.encode(), dtype=np.uint8)
    ey = np.frombuffer(y.encode(), dtype=np.uint8)
    ez = np.frombuffer(z.encode(), dtype=np.uint8)

    D, ptr = _fill(ex, ey, ez)
    score = int(D[len(x), len(y), len(z)])
    moves = _traceback(ptr, len(x), len(y), len(z))

    rows = [[], [], []]
    path = [(0, 0, 0)]
    i = j = l = 0
    for m in moves:
        a, b, d = MA[m], MB[m], MD[m]
        rows[0].append(x[i] if a else "-")
        rows[1].append(y[j] if b else "-")
        rows[2].append(z[l] if d else "-")
        i, j, l = i + a, j + b, l + d
        path.append((i, j, l))

    return score, ["".join(r) for r in rows], path


def dp3_memory_gb(lx, ly, lz):
    """Memory needed for a DP of these lengths, in GB."""
    return 5 * (lx + 1) * (ly + 1) * (lz + 1) / 1e9
