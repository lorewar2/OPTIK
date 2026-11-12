import random

random.seed(42)
BASES = "ACGT"

def generate_set(set_id, length, n_copies=3, sub_rate=0.01, ins_rate=0.005, del_rate=0.005):
    """Make one original sequence and n_copies mutated copies of it."""
    original = "".join(random.choice(BASES) for _ in range(length))
    records = [(f"{set_id}-original", original)]

    for c in range(1, n_copies + 1):
        out = []
        for base in original:
            r = random.random()
            if r < del_rate:                          # deletion: drop this base
                continue
            if r < del_rate + ins_rate:               # insertion: add a random base before it
                out.append(random.choice(BASES))
            if random.random() < sub_rate:            # substitution
                base = random.choice([b for b in BASES if b != base])
            out.append(base)
        records.append((f"{set_id}-{c}", "".join(out)))

    return records


def write_fasta(filename, records):
    """Write (name, sequence) records to a FASTA file, with each sequence's length in its header."""
    with open(filename, "w") as f:
        for name, seq in records:
            f.write(f">{name} length={len(seq)}\n{seq}\n")

records = []
number_of_sequences = 100
for set_id in range(1, 1 + number_of_sequences):
    length = random.randint(20, 2_000)
    records += generate_set(set_id, length)
write_fasta("./data/sequences.fastq", records)