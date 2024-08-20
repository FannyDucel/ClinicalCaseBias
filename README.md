# Gender biases in generated clinical cases
This rep gathers all the codes, data and results from the Clinical Cases project. Its goal is to automatically generate 
clinical cases in French, with fine-tuned LLMs on 10 pathologies that are (or not) stereotypically/statistically associated with a gender (feminine or masculine).
Then, we use an automatic gender detection system to automatically annotate the gender of the patient that was generated in the clinical case.
That way, we can compute gender biases based on the differences of generated gender (e.g. are there way more masculine (vs. feminine) patients generated for texts about bladder cancer?)

# Repo organization
- annotated_data/: contains generated clinical cases. All but corpus_manual_annote have their gender automatically annotated.
- bias_results/: contains some csv files and figures (to update) based on results obtained after bias metrics computation. Also contains analyse_phrase1.ipynb to look at the data and some other demographic features.
- generated_data/: contains the generated clinical cases in raw json files (in corresponding subfolder raw_json/) and their DataFrames versions
- ressources_lgq/: contains two json files with lists of epicene nouns as well as nouns referring to human entities
- src/: 
  - bias_exploration/:
    - correlation_bias-contraintes.py: a file to compute some correlations on data
    - measure_bias.py: to compute Gender Gaps, Gender Shifts and some other stats
    - visualize_total.ipynb: notebook to generate some figures on the entire corpus
    - visualize_neutralprompts.ipynb: notebook to generate the same figures as _total but only on generations that answer a neutral prompt
  - preparation/:
    - evaluation.py: to evaluate the gender detection system based on manual annotations #TODO
    - gender_detection_adaptation: the gender detection system that has been adapted for P3 (and for the medical domain) from [[https://inria.hal.science/hal-04621134/]]
    - preprocess_files: to convert generations from json to csv
    - repetition.py: to compute repetitions rates in generations (to select better corpus)
  - main.py : #TODO

# How to replicate the experiments
## Fine-tune your LLMs
TODO

## Generate your clinical cases
TODO

## Evaluate bias

1. preprocess_files.py
2. gender_detection_adaptation.py
3. evaluation.py (if manual annotations)
4. compute stats (measure_bias.py, ipynb for visualization + correlation_bias-contraintes et repetition)