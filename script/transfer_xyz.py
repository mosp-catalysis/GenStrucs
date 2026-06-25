import pickle
import sys
import os
import numpy as np
from ase import Atoms

# Make CatGPT utilities importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import PreTrainedTokenizerFast
from catgpt.utils.generation_utils import str_to_atoms  # used only to obtain lattice cell


pkl_path = "/home/zbe/gene_CH3/gene_50w_16/gene500w_16.pkl"
tokenizer_path = "/home/zbe/data/tokenizer/coordinate-tokenizer/"

# Load tokenizer

with open(pkl_path, "rb") as f:
    gens = pickle.load(f)

tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


def parse_atoms_manual(atoms_str, cell):
    toks = atoms_str.split()
    elems = []
    fracs = []

    i = 0
    n = len(toks)
    while i < n:
        t = toks[i]

        # Skip special tokens such as <sep>, <lat>, etc.
        if t.startswith("<") and t.endswith(">"):
            i += 1
            continue

        if t.isalpha() and len(t) <= 2:
            if i + 3 >= n:
                break

            try:
                x = float(toks[i + 1])
                y = float(toks[i + 2])
                z = float(toks[i + 3])
            except (ValueError, IndexError):
                # Coordinates are invalid or out of bounds
                i += 1
                continue

            elems.append(t)
            fracs.append([x, y, z])
            i += 4
            continue

        i += 1

    if not elems:
        return None

    fracs = np.array(fracs, dtype=float)
    atoms = Atoms(symbols=elems, cell=cell, pbc=True)
    atoms.set_scaled_positions(fracs)

    return atoms


bad_idx = []
bad_reason = []

for idx, seq in enumerate(gens[:500000]):
    ids = seq[0].tolist()
    text = tokenizer.decode(ids, skip_special_tokens=False)

    if "<eos>" in text:
        text = text.split("<eos>")[0]

    if text.startswith("<bos>"):
        text = text.replace("<bos>", "", 1).strip()

    atoms_str = text.strip()

    tmp_atoms, struct_val, gen_val = str_to_atoms(
        atoms_str,
        lat_idx=0,
        skip_fail=False,
        early_stop=False,
    )

    if tmp_atoms is None:
        bad_idx.append(idx)
        bad_reason.append(f"str_to_atoms failed (struct_val={struct_val}, gen_val={gen_val})")
        print(f"[SKIP] Sample {idx}: str_to_atoms failed")
        continue

    cell = tmp_atoms.get_cell()

    manual_atoms = parse_atoms_manual(atoms_str, cell)
    if manual_atoms is None:
        bad_idx.append(idx)
        bad_reason.append("manual parsing failed (no valid atoms)")
        print(f"[SKIP] Sample {idx}: manual parsing failed")
        continue

    # Write XYZ file
    manual_atoms.write(f"sample_{idx}.xyz")


with open("bad_samples.txt", "w") as fw:
    for i, r in zip(bad_idx, bad_reason):
        fw.write(f"{i}\t{r}\n")

print(f"[DONE] bad samples: {len(bad_idx)} (saved to bad_samples.txt)")
