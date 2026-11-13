import random

random.seed(42)
BASES = "ACGT"
GC = 0.41                                   # human genome average GC content

# PacBio-like error profiles (per base)
#   subread: raw PacBio subreads/CLR, ~10% error, mostly insertions
PROFILES = {
    "subread": dict(sub=0.010,  ins=0.060,  dele=0.030,  hp=0.020),
}
PROFILE_WEIGHTS = {"hifi": 0.5, "subread": 0.5}    # share of sets using each profile


def random_bases(n):
    """Random DNA with genome-like GC content."""
    at, cg = (1 - GC) / 2, GC / 2
    return "".join(random.choices(BASES, weights=[at, cg, cg, at], k=n))


def substitute(seq, rate):
    return "".join(random.choice([c for c in BASES if c != b]) if random.random() < rate else b
                   for b in seq)


def make_original(length):
    """
    Genome-like original sequence built from segments:
      - random sequence (most of it)
      - tandem repeats (1-6 bp unit repeated 3-15 times), including homopolymers
      - duplications of earlier parts of the sequence (with ~5% divergence)
    """
    seq = ""
    while len(seq) < length:
        r = random.random()
        if r < 0.10:                                    # tandem repeat / homopolymer
            unit = random_bases(random.randint(1, 6))
            seq += unit * random.randint(3, 15)
        elif r < 0.15 and len(seq) > 50:                # duplication of an earlier segment
            n = random.randint(20, min(200, len(seq)))
            start = random.randint(0, len(seq) - n)
            seq += substitute(seq[start:start + n], 0.05)
        else:                                           # unique sequence
            seq += random_bases(random.randint(20, 100))
    return seq[:length]


def add_errors(seq, sub, ins, dele, hp):
    """
    PacBio-like read errors:
      - homopolymer runs (>= 3) gain or lose a base with probability hp * run length
      - insertions, half of which repeat the current base (PacBio insertions favor this)
      - deletions and substitutions
    """
    out = []
    i = 0
    while i < len(seq):
        b = seq[i]
        j = i
        while j < len(seq) and seq[j] == b:
            j += 1
        run = j - i
        if run >= 3 and random.random() < hp * run:     # homopolymer length error
            run += random.choice([-1, 1])
        for _ in range(run):
            r = random.random()
            if r < dele:                                # deletion
                continue
            if r < dele + ins:                          # insertion before this base
                out.append(b if random.random() < 0.5 else random.choice(BASES))
            base = b
            if random.random() < sub:                   # substitution
                base = random.choice([c for c in BASES if c != b])
            out.append(base)
        i = j
    return "".join(out)


def generate_set(set_id, length, n_copies=3):
    """One genome-like original and n_copies PacBio-like reads of it."""
    profile = "subread"
    rates = PROFILES[profile]

    original = make_original(length)
    records = [(f"{set_id}-original", original, profile)]

    for c in range(1, n_copies + 1):
        jitter = random.uniform(0.5, 1.5)               # read-to-read quality variation
        read = add_errors(original, **{k: v * jitter for k, v in rates.items()})
        records.append((f"{set_id}-{c}", read, profile))

    return records


def write_fasta(filename, records):
    """Write records to FASTA, with length and error profile in the header."""
    with open(filename, "w") as f:
        for name, seq, profile in records:
            f.write(f">{name} length={len(seq)} profile={profile}\n{seq}\n")

records = []
number_of_sequences = 10_000
for set_id in range(1, 1 + number_of_sequences):
    length = random.randint(20, 500)
    records += generate_set(set_id, length)
write_fasta("./data/sequences.fasta", records)