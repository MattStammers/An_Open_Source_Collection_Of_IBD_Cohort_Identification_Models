"""

Filter_MRCONZO_ibd_drugs.py

This script filters MRCONZO drug names so that they are relevant to IBD.

This is to reduce the computation time for UMLS mapping and to prevent inappropriate terms being mapped

"""

import csv
import os
import re

# Import configuration constants
import constants as c
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# IBD Drug Filter MRCONZO For UMLS                                            #
# --------------------------------------------------------------------------- #

def filter_mrconso_for_ibd_drugs(
    umls_dir,
    ibd_regexes,
    drug_regexes,
    output_filename="MRCONSO_filtered.RRF",
    keep_only_english=True,
):
    mrconso_path = os.path.join(umls_dir, "MRCONSO.RRF")
    filtered_path = os.path.join(umls_dir, output_filename)

    if not os.path.exists(mrconso_path):
        raise FileNotFoundError(f"MRCONSO.RRF not found in {umls_dir}")

    combined_pattern = re.compile(
        r"|".join(f"(?:{pat})" for pat in ibd_regexes + drug_regexes),
        flags=re.IGNORECASE,
    )

    # ── FIRST PASS: collect CUIs, tracking TTY drops
    relevant_cuis = set()
    total_count = 0
    tty_dropped_fp = 0
    with open(mrconso_path, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile, delimiter="|")
        for row in tqdm(reader, desc="First pass: collecting CUIs"):
            total_count += 1
            if len(row) < 15:
                continue
            cui, lat, tty, term_str = row[0], row[1], row[12], row[14]
            if keep_only_english and lat != "ENG":
                continue
            if tty not in ("PT", "SY"):
                tty_dropped_fp += 1
                continue
            if combined_pattern.search(term_str):
                relevant_cuis.add(cui)
    print(
        f"First pass done: {total_count:,} lines, dropped {tty_dropped_fp:,} by TTY filter, found {len(relevant_cuis):,} CUIs."
    )

    # ── SECOND PASS: write rows, tracking TTY drops again
    kept_count = 0
    total_count_2 = 0
    tty_dropped_sp = 0
    with open(mrconso_path, "r", encoding="utf-8") as infile, open(
        filtered_path, "w", encoding="utf-8", newline=""
    ) as outfile:

        reader = csv.reader(infile, delimiter="|")
        writer = csv.writer(outfile, delimiter="|", lineterminator="\n")

        for row in tqdm(reader, desc="Second pass: writing relevant rows"):
            total_count_2 += 1
            if len(row) < 15:
                continue
            cui, lat, tty = row[0], row[1], row[12]
            if keep_only_english and lat != "ENG":
                continue
            if tty not in ("PT", "SY"):
                tty_dropped_sp += 1
                continue
            if cui in relevant_cuis:
                writer.writerow(row)
                kept_count += 1

    print(
        f"Second pass done: {total_count_2:,} lines, dropped {tty_dropped_sp:,} by TTY filter, wrote {kept_count:,} rows to {filtered_path}"
    )

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    umls_root = os.path.abspath(
        os.path.join(script_dir, "..", "..", "..", "data", "umls", "AB_full", "META")
    )
    filter_mrconso_for_ibd_drugs(
        umls_root, ibd_regexes=c.IBD_KEYWORDS, drug_regexes=c.DRUG_KEYWORDS
    )
