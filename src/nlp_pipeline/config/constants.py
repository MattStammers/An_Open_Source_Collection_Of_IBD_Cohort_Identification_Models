"""

Constants.py

This file describes all the main constants used in the module providing routes for all the analytics.

It also provides some mapping functions which are reused and do not change during the project.

"""

import os

# --------------------------------------------------------------------------- #
# All Core Data Paths                                                         #
# --------------------------------------------------------------------------- #

try:
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
except NameError:
    ROOT_DIR = os.getcwd()
print("Project ROOT_DIR:", ROOT_DIR)

# Data paths
DATA_DIR = os.path.join(ROOT_DIR, "data", "document_validation")
TRAIN_FILE_PATH = os.path.join(DATA_DIR, "merged_human_train_redacted.xlsx")
VAL_FILE_PATH = os.path.join(DATA_DIR, "merged_human_validation_redacted.xlsx")
TRAIN_FILE_UMLS_PATH = os.path.join(DATA_DIR, "UMLS_standardised_train.xlsx")
VAL_FILE_UMLS_PATH = os.path.join(DATA_DIR, "UMLS_standardised_val.xlsx")

# Output directories
REGEX_ANALYSIS_DIR = os.path.join(ROOT_DIR, "data", "results", "regex_analysis")
SPACY_ANALYSIS_DIR = os.path.join(ROOT_DIR, "data", "results", "spacy_analysis")
BOW_ANALYSIS_DIR = os.path.join(ROOT_DIR, "data", "results", "bow_analysis")
TFIDF_ANALYSIS_DIR = os.path.join(ROOT_DIR, "data", "results", "tfidf_analysis")
WORD2VEC_ANALYSIS_DIR = os.path.join(ROOT_DIR, "data", "results", "word2vec_analysis")
SBERT_ANALYSIS_DIR = os.path.join(ROOT_DIR, "data", "results", "sbert_base_analysis")
SBERT_MED_ANALYSIS_DIR = os.path.join(ROOT_DIR, "data", "results", "sbert_med_analysis")
DISTILBERT_ANALYSIS_DIR = os.path.join(
    ROOT_DIR, "data", "results", "distilbert_analysis"
)
BIO_CLINICAL_BERT_ANALYSIS_DIR = os.path.join(
    ROOT_DIR, "data", "results", "bio_clinical_bert_analysis"
)
ROBERTA_ANALYSIS_DIR = os.path.join(ROOT_DIR, "data", "results", "roberta_analysis")
LLM_ANALYSIS_DIR = os.path.join(ROOT_DIR, "data", "results", "llm_analysis")
FINAL_RESULTS_DIR = os.path.join(ROOT_DIR, "nlp_final_model_results")

# UMLS directory
UMLS_DIR = os.path.join(ROOT_DIR, "data", "umls", "AB_full", "META")

# Caching directory
CACHE_DIR = os.path.join(ROOT_DIR, "cache_dir")

# Ensure these directories exist
for directory in [
    REGEX_ANALYSIS_DIR,
    SPACY_ANALYSIS_DIR,
    BOW_ANALYSIS_DIR,
    FINAL_RESULTS_DIR,
    TFIDF_ANALYSIS_DIR,
    WORD2VEC_ANALYSIS_DIR,
    SBERT_ANALYSIS_DIR,
    SBERT_MED_ANALYSIS_DIR,
    DISTILBERT_ANALYSIS_DIR,
    BIO_CLINICAL_BERT_ANALYSIS_DIR,
    ROBERTA_ANALYSIS_DIR,
    LLM_ANALYSIS_DIR,
    CACHE_DIR,
]:
    os.makedirs(directory, exist_ok=True)

# --------------------------------------------------------------------------- #
# Core Required Data Manipulations                                            #
# --------------------------------------------------------------------------- #

# Columns to drop
COLUMNS_TO_DROP = [
    "IBD_Prediction",
    "m42_Full_Response_preceding_clinic_zero_clinic_endo_hist",
    "m42_Json_Response_preceding_clinic_zero_clinic_endo_hist",
    "m42_Payload_preceding_clinic_zero_clinic_endo_hist",
    "Truncated_preceding_clinic_zero_clinic_endo_hist",
]

# For per-report IBD classification
COLWISE_COLUMNS = {
    "histopathology_report": "Histopath_IBD",
    "endoscopy_report": "Endoscopy_IBD",
    "preceding_clinic_letter": "Preceding_Clinic_IBD",
    "following_clinic_letter": "Following_Clinic_IBD",
}

# For patient-level classification
PATIENT_COLUMNS = {
    "histopathology_report": "Patient_Has_IBD",
    "endoscopy_report": "Patient_Has_IBD",
    "preceding_clinic_letter": "Patient_Has_IBD",
    "following_clinic_letter": "Patient_Has_IBD",
}

# IBD columns mapping (gold-standard labels)
IBD_COLUMNS = {
    "histopathology_report": "Histopath_IBD",
    "endoscopy_report": "Endoscopy_IBD",
    "preceding_clinic_letter": "Preceding_Clinic_IBD",
    "following_clinic_letter": "Following_Clinic_IBD",
}

# DRUG_COLUMNS (gold standard labels)
DRUG_COLUMNS = {
    "histopathology_report": "Histopath_IBD",
    "endoscopy_report": "Endoscopy_IBD",
    "preceding_clinic_letter": "Preceding_Clinic_IBD",
    "following_clinic_letter": "Following_Clinic_IBD",
}

# --------------------------------------------------------------------------- #
# Keywords                                                                    #
# --------------------------------------------------------------------------- #

# Regex keywords for IBD and Drugs
IBD_KEYWORDS = [
    r"\bibd\b",
    r"\binflammatory bowel disease\b",
    r"\bcrohn'?s\b",
    r"\bcolitis\b",
    r"\bulcerative colitis\b",
]

DRUG_KEYWORDS = [
    r"\bhydrocortisone\b",
    r"\bprednisolone\b",
    r"\boctasa\b",
    r"\bsalofalk\b",
    r"\bpentasa\b",
    r"\basacol\b",
    r"\bmesalazine\b",
    r"\bbudenofalk\b",
    r"\bbudesonide\b",
    r"\bsulfasalazine\b",
    r"\bsalazopyrin\b",
    r"\btofacitinib\b",
    r"\badalimumab\b",
    r"\bmethotrexate\b",
    r"\betanercept\b",
    r"\bazathioprine\b",
    r"\bmercaptopurine\b",
    r"\bustekinumab\b",
    r"\binfliximab\b",
    r"\bvedolizumab\b",
    r"\bmirikizumab\b",
    r"\bupadacitinib\b",
    r"\bfilgotinib\b",
]

CUI_KEYWORDS = [r"\(C\d{4,}\)"]

# --------------------------------------------------------------------------- #
# Demographic and Negation Keys                                               #
# --------------------------------------------------------------------------- #

# Demographic attributes to check in the data
DEMOGRAPHICS_KEYS = ["gender", "ethnicity", "imd_group", "age_group"]

# Negation termset configuration for NegSpacy
NEG_TERMSET = {
    "pseudo_negations": [
        "without",
        "no",
        "negative for",
        "rule out",
        "exclude",
        "absence of",
    ],
    "preceding_negations": [
        "not",
        "no evidence of",
        "without",
    ],
    "following_negations": [
        "no evidence of",
        "no signs of",
        "no history of",
        "no indication of",
    ],
    "termination": [";", "."],
}

# --------------------------------------------------------------------------- #
# Legacy Dictionaries                                                         #
# --------------------------------------------------------------------------- #

# Prediction types mapping (for each report column, which prediction types to run)
PREDICTION_TYPES = {
    "histopathology_report": ["IBD", "Drug", "Combined"],
    "endoscopy_report": ["IBD", "Drug", "Combined"],
    "preceding_clinic_letter": ["IBD", "Drug", "Combined"],
    "following_clinic_letter": ["IBD", "Drug", "Combined"],
}