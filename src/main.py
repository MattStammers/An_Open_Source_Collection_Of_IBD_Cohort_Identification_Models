#!/usr/bin/env python3
"""

This script runs the entire NLP workflow for the project and acts as an entrypoint.

A --test flag triggers pytest with or without warnings. 

"""
import argparse
import os
import subprocess
import sys

# Get to the Root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nlp_pipeline.models.bio_clinical_bert.pipeline import (
    main as bio_clinical_bert_main,
)
from nlp_pipeline.models.bow.pipeline import main as bow_main
from nlp_pipeline.models.distilbert.pipeline import main as distilbert_main
from nlp_pipeline.models.regex.pipeline import main as regex_main
from nlp_pipeline.models.roberta.pipeline import main as roberta_main
from nlp_pipeline.models.sbert_base.pipeline import main as sbert_main
from nlp_pipeline.models.sbert_med.pipeline import main as sbert_med
from nlp_pipeline.models.spacy.pipeline import main as spacy_main
from nlp_pipeline.models.tf_idf.pipeline import main as tfidf_main
from nlp_pipeline.models.word2vec.pipeline import main as word2vec_main

# --------------------------------------------------------------------------- #
# Runs the Full Pipeline                                                      #
# --------------------------------------------------------------------------- #

def run_tests():
    raw = sys.argv[1:]

    disable_warnings = False
    enable_warnings = False
    if "--disable-warnings" in raw or "--no-warnings" in raw:
        disable_warnings = True
    if "--with-warnings" in raw:
        enable_warnings = True

    control = {
        "--test",
        "--disable-warnings",
        "--no-warnings",
        "--with-warnings",
    }
    passed = [a for a in raw if a not in control]

    opts = [a for a in passed if a.startswith("-")]
    paths = [a for a in passed if not a.startswith("-")]

    if not paths:
        paths = ["tests/"]

    cmd = [sys.executable, "-m", "pytest"]

    cmd += [
        "-q", 
        "-o",
        "log_cli=true",  
        "-o",
        "log_cli_level=WARNING", 
        "-o",
        "log_level=WARNING", 
    ]

    cmd += opts + paths

    ret = subprocess.call(cmd)
    sys.exit(ret)

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--disable-umls",
        action="store_true",
        help="Skip all UMLS-standardised variants",
    )
    args, _ = parser.parse_known_args()

    if args.test:
        run_tests()

    regex_main(disable_umls=args.disable_umls)
    spacy_main(disable_umls=args.disable_umls)
    bow_main(disable_umls=args.disable_umls)
    tfidf_main(disable_umls=args.disable_umls)
    word2vec_main(disable_umls=args.disable_umls)
    sbert_main(disable_umls=args.disable_umls)
    sbert_med(disable_umls=args.disable_umls)
    distilbert_main(disable_umls=args.disable_umls)
    bio_clinical_bert_main(disable_umls=args.disable_umls)
    roberta_main(disable_umls=args.disable_umls)

if __name__ == "__main__":
    main()