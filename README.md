# Open-Source IBD Cohort Identification Models

### Subtitle: IBD_NLP_Cohort_Identification_Models_IC-IBD_Part_2

## By Dr Matt Stammers

### Completed: 22/06/2025. Readme updated: 30/05/2026.

## Purpose

This repository contains the code used to train, evaluate, and analyse a suite of open-source NLP pipelines for inflammatory bowel disease cohort identification. Its primary purpose is to support transparency, methodological review, and reproducibility in an academic setting.

The repository includes the non-LLM pipelines used for the main comparative modelling work reported in Chapter 3 of the thesis. It should be treated as a research artefact rather than a production clinical system.

## Associated Publications

1. Stammers M, Gwiggner M, Nouraei R, Metcalf C, Batchelor J. Robust comparative evaluation of 15 natural language processing algorithms to positively identify patients with inflammatory bowel disease from secondary care records. BMJ Open Gastroenterology. 2025;12(1). [BMJ Open Gastroenterology](https://bmjopengastro.bmj.com/content/12/1/e001977)
2. Stammers M, Sartain S, Cummings JF, Kipps C, Nouraei R, Gwiggner M, Metcalf C, Batchelor J. Identification of cohorts with inflammatory bowel disease amidst fragmented clinical databases via machine learning. Digestive Diseases and Sciences. 2025;70(10):3309-22. [Digestive Diseases and Sciences](https://link.springer.com/article/10.1007/s10620-025-09323-1)
3. Stammers M, Ramgopal B, Owusu Nimako A, Vyas A, Nouraei R, Metcalf C, Batchelor J, Shepherd J, Gwiggner M. A foundation systematic review of natural language processing applied to gastroenterology and hepatology. BMC Gastroenterology. 2025;25(1):58. [BMC Gastroenterology](https://bmcgastroenterol.biomedcentral.com/articles/10.1186/s12876-025-03608-5)

## Repository Scope

This repository includes:

- rule-based pipelines: regex and spaCy
- classical machine learning pipelines: bag-of-words, TF-IDF, and Word2Vec
- fine-tuned transformer pipelines: SBERT-base, SBERT-med, DistilBERT, BioClinicalBERT, and RoBERTa
- shared utilities for evaluation, fairness, explainability, logging, caching, and resource monitoring
- automated tests for pipelines and utilities
- selected released model artefacts under `models/`

This repository does not include the full LLM serving and prompt-execution infrastructure used for chapter four - this is provided in 
https://github.com/MattStammers/FM_metacognition_during_cohort_identification. The local LLM code present here is limited to downstream parsing and analytics of generated outputs.

## Repository Structure

The codebase is organised around reusable shared utilities plus model-specific workflow packages.

```text
An_Open_Source_Collection_Of_IBD_Cohort_Identification_Models/
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- src/
|   |-- main.py
|   `-- nlp_pipeline/
|       |-- common/
|       |   |-- caching.py
|       |   |-- data_utils.py
|       |   |-- evaluation.py
|       |   |-- fairness.py
|       |   |-- feature_importance.py
|       |   |-- logging_setup.py
|       |   `-- resource_monitor.py
|       |-- config/
|       |   |-- constants.py
|       |   `-- sample_size_calculation.py
|       |-- doc2patient/
|       |   |-- decision_tree.py
|       |   |-- heatmap.py
|       |   `-- log_reg.py
|       |-- models/
|       |   |-- regex/
|       |   |   |-- model.py
|       |   |   `-- pipeline.py
|       |   |-- spacy/
|       |   |   |-- model.py
|       |   |   `-- pipeline.py
|       |   |-- bow/
|       |   |   |-- model.py
|       |   |   `-- pipeline.py
|       |   |-- tf_idf/
|       |   |   |-- model.py
|       |   |   `-- pipeline.py
|       |   |-- word2vec/
|       |   |   |-- model.py
|       |   |   `-- pipeline.py
|       |   |-- sbert_base/
|       |   |-- sbert_med/
|       |   |-- distilbert/
|       |   |-- bio_clinical_bert/
|       |   |-- roberta/
|       |   `-- llm_analytics/
|       `-- tests/
|           |-- utilities/
|           |-- bow/
|           |-- tfidf/
|           |-- word2vec/
|           |-- spacy/
|           |-- regex/
|           |-- sbert/
|           |-- sbert_med/
|           |-- distilbert/
|           `-- roberta/
|-- models/
|   |-- bert/
|   |-- bow/
|   |-- regex/
|   |-- spacy/
|   |-- tfidf/
|   `-- word2vec/
`-- docs/
```

The inline tree above provides a high-level map of the repository layout for readers who want an explicit structural summary before working through the detailed description below.

- `src/main.py`: top-level entry point that runs the included pipelines sequentially
- `src/nlp_pipeline/config/`: global constants, file paths, and column mappings
- `src/nlp_pipeline/common/`: shared utilities for data loading, evaluation, fairness, explainability, caching, logging, and resource monitoring
- `src/nlp_pipeline/models/`: one subpackage per model family
- `src/nlp_pipeline/doc2patient/`: supporting analyses focused on document-to-patient relationships
- `src/tests/`: automated tests for models and shared utilities
- `models/`: selected released model artefacts and model-specific notes

Within `src/nlp_pipeline/models/`, most model families follow the same internal structure:

- `pipeline.py`: orchestration of the full workflow for that model family
- `model.py`: training, prediction, and model-specific helper logic
- `__init__.py`: package definition

This means the repository is not built around one single pipeline file. Instead, it uses a repeated experimental framework applied to multiple model families.

## How Execution Works

Running `python src/main.py` triggers a sequence of independent model pipelines rather than one combined ensemble model.

At present the main script executes the following in order:

1. regex
2. spaCy
3. bag-of-words
4. TF-IDF
5. Word2Vec
6. SBERT-base
7. SBERT-med
8. DistilBERT
9. BioClinicalBERT
10. RoBERTa

Each of those model families exposes its own `main()` function, and each `main()` function is responsible for running that model family end to end.

## How a Typical Pipeline Works

Although the implementation differs slightly by model family, the workflows are intentionally similar so that outputs can be compared more directly.

### 1. Data loading and preprocessing

Each pipeline starts by loading training and validation data through the shared preprocessing code in `common/data_utils.py`.

This stage typically:

- reads CSV or Excel input files
- removes configured columns that are not required
- checks expected text columns and label columns
- standardises demographic fields used for subgroup analysis
- prepares combined text fields where required
- ensures patient-level labels such as `Patient_Has_IBD` are available when needed downstream

### 2. Split validation and leakage checks

Before fitting a model, pipelines usually run simple diagnostics intended to identify obvious leakage problems. These checks commonly include patient overlap between train and validation sets and exact duplicate document content across splits.

### 3. Document-level training

The central pattern in this repository is that model training starts at the document level. For each configured report column, a model is trained to classify that specific document type.

Depending on the model family, this may involve:

- a rule-based matcher
- a vectoriser plus logistic regression classifier
- a transformer fine-tuning workflow

### 4. Document-level prediction and evaluation

After training, the pipeline writes predictions and probabilities back onto the data frame and evaluates them using the shared metric engine in `common/evaluation.py`.

This evaluation layer is responsible for metrics such as:

- accuracy
- precision
- recall
- specificity
- NPV
- F1 score
- MCC
- Brier score and calibration outputs where available

### 5. Aggregation from documents to patients

After document-level predictions are produced, the pipelines aggregate those outputs to patient level, usually by grouping on `study_id` and collapsing predictions across a patient's available documents.

This is a key design feature of the repository because the thesis analyses compare document-level behaviour with patient-level cohort identification.

### 6. Patient-level evaluation

The patient-level outputs are then evaluated against patient-level gold labels. This provides the patient-level summary metrics used in the downstream analyses and tables.

### 7. Additional artefacts

Depending on the pipeline, extra outputs may then be generated, such as:

- calibration summaries
- feature-importance files
- SHAP or LIME explainability outputs
- fairness tables and plots
- runtime and emissions summaries
- saved models in joblib or transformer formats

## Shared Versus Model-Specific Code

Understanding this separation is useful when reviewing the repository.

Shared code under `src/nlp_pipeline/common/` handles tasks that are reused across model families, including:

- preprocessing support
- metric calculation
- fairness analysis
- plotting
- logging
- monitoring and emissions tracking

Model-specific code under `src/nlp_pipeline/models/` contains the parts that differ between approaches, such as:

- regex rules
- spaCy phrase matching logic
- scikit-learn vectorisation and classifier setup
- transformer training and inference code

In short, `pipeline.py` files define the experimental workflow, while `model.py` files define the modelling mechanics.

## External Model Links

- Hugging Face collection: [BERT-based IBD models](https://huggingface.co/collections/MattStammers/a-collection-of-ibd-bert-models-682b01badbaa646380f54b14)
- Hugging Face demo: [IBD Cohort Identification Demo](https://huggingface.co/spaces/MattStammers/IBD_Cohort_Identification)

## Ratings/Features
- Python Difficulty Level: Fairly Advanced (Not Particularly Recommended for Beginners)
- Primary Code Purpose: Code Your Own Versions. Transparency for paper. Maximising generalisability and replicability.

### How to Use Yourself

To run the code you will have to appropriately prepare your (ideally poetry environment) study_id's and string data into seperated columns in a dataframe. I recommend using .py files rather than .ipynb notebooks for this but the choice is up to you and will to some degree depend upon level of experience. For a basic primer on using python and setting it up for the first time: [Python Starter Guide](https://mattstammers.github.io/hdruk_avoidable_admissions_collaboration_docs/how_to_guides/new_to_python)

Analysts must prepare the environment appropriately. I have written a guide before which I will link into this repo. Alternatively, if you are new to python and working in a healthcare context I recommend visiting for a basic-advanced quick into: [NHS BI Analyts Python for Data Science Intro](https://github.com/MattStammers/Community_Of_Practice_Session_Two)

### How to Use Yourself

1. Install environments.

The first thing to flag is that this pipeline works best in Linux environments. It does run in Windows but less successfully. All Windows dependencies have been removed to make it interoperable.

The recommendation is to use poetry to install a cuda enabled environment otherwise the pipeline will take a long time to run. This can be achieved as follows:

```sh
pip install poetry
poetry install --extras "cuda"
pip install -r requirements.txt
```

If a CPU-only environment is required, dependency resolution may need to be adapted locally. The repository was developed for a research workflow rather than a minimal deployment footprint.

## Running the Test Suite

From the repository root:

```sh
python src/main.py --test
```

Optional pytest flags can be passed through. For example:

```sh
python src/main.py --test -q
python src/main.py --test --disable-warnings
```

The configured pytest path is `src/tests`.

## Running the Pipelines

From the repository root:

```sh
python src/main.py
```

To skip the UMLS-standardised variants:

```sh
python src/main.py --disable-umls
```

The main entry point runs the included non-LLM and fine-tuned transformer pipelines sequentially.

## Outputs

Depending on the pipeline and configuration, generated outputs may include:

- evaluation tables
- calibration summaries
- fairness reports
- feature-importance artefacts
- SHAP or LIME outputs where enabled
- emissions and resource-monitoring summaries
- saved model artefacts

Default output locations are configured centrally in `src/nlp_pipeline/config/constants.py`.

## Important Limitations

These models are research artefacts and should be interpreted accordingly.

- They were developed and evaluated on a single-site research dataset.
- They may encode biases present in the source cohort and documentation practices.
- Performance may not generalise to other institutions, EHR systems, or patient populations.
- Some pipelines are computationally expensive and require GPU support.
- The released models are not validated for clinical deployment and should not be treated as medical devices.

## Intended Use

This repository is suitable for:

- thesis review and methodological inspection
- reproducibility-oriented academic work
- comparative NLP research
- further model development in governed research environments

It is not intended for direct clinical decision-making or unsupervised operational deployment.

## Corrections

30/05/2026: Tidied BoW pipeline to match other pipelines. Directionality error in EO recognised. Doc2Patient inconsistency in SBERT pipelines recognised but did not impact final results so left as it was.

## Contributing

Contributions that improve clarity, reproducibility, testing, or methodological robustness are welcome. Please submit a pull request with a clear description of the proposed change.

## Licence

This project is released under the Creative Commons Attribution-NonCommercial 4.0 International licence.

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

## Legal and Governance Notice

No warranty is provided regarding performance, safety, or fitness for a particular purpose. These models are not CE-marked medical devices and should be used only within appropriate research and governance frameworks. Users are responsible for ensuring lawful and ethical use, including compliance with applicable information governance, data protection, and regulatory requirements.
