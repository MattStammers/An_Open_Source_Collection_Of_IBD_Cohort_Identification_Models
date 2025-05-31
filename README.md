# An Open Source Collection Of IBD Cohort Identification Models
### IBD_NLP_Cohort_Identification_IC-IBD_Study_Part_2

## By Matt Stammers

### 31/05/2025

IBD Cohort Identification Code To Go with Accompanying Papers and Models

These artefacts are made freely available to researchers to promote open science, improve robustness and make the work open to scrutiny. If a mistake or issue is found please raise an issue on the repository.

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
cd src
poetry env activate
poetry install --extras "cuda"
```

This installs all the base packages. However, to complete the process it is easier to use pip to complete the installation. If you find a good way to do it all with poetry please send me a pull request.

```sh
pip install -r requirements.txt
```

2. Run Tests.

Before you run the pipeline it is recommended to run the test suite. This can be achieved with the following if you want to see the warnings as well:

```sh
python main.py --test --disable-warnings --capture=tee-sys -rw
```

If all the tests succeed then it is likely the pipeline will run successfully.

3. Run The Pipeline

Now you are ready to run the full pipeline (providing you have sufficient VRAM and compute available). If not either shift everything to the CPU or configure accelerate to load balance the training process.

To run the pipeline run:

```sh
python main.py --disable-umls
```

### Contributing

If you would like to contribute futher to this project you can do so by submitting a pull request to this repo. If you remix or fork the project please attribute appropraitely. These models should not be used commercially. Obtaining profit by using them is forbidden as per the licence. If they are improved then they must be shared open source to the community.

### Licence

This project and the associated models are Attribution-NonCommercial 4.0 International Licensed. The copyright holder is Matt Stammers and University Hospital Southampton NHS Foundation Trust.

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

### Disclaimer

No guarantee is given of model performance in any production capacity whatsoever. These models should be used in full accordance with the EU AI Act - Regulation 2024/1689. These are not CE marked medical devices and are suitable at this point only for research and development / experimentation. They can be improved but any improvements should be published openly and shared openly with the community. UHSFT and the author own the copyright and are choosing to share them freely under a CC BY-NC 4.0 Licence for the benefit of the wider research community.
