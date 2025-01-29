# "Women do not have heart attacks!" Gender Biases in Automatically Generated Clinical Cases in French
This repo gathers all the code, data and results from the study that led to the article "Women do not have heart attacks!"
Gender Biases in Automatically Generated Clinical Cases in French (Ducel et al., 2025), published at NAACL Findings. 

Its goal is to automatically generate clinical cases in French, with 7 fine-tuned LLMs on 10 pathologies that are (or not)
stereotypically/statistically associated with a gender (feminine or masculine).
Then, we use an automatic gender detection system to automatically annotate the gender of the patient that was generated 
in the clinical case.
That way, we can compute gender biases based on the differences of generated gender 
(e.g. are there way more masculine (vs. feminine) patients generated for texts about bladder cancer?).

With this repo, you can reproduce our experiments or adapt our code to extend the scope of our study. 

If you have questions, feel free to reach out: *fanny.ducel@universite-paris-saclay.fr*

# Repo organization
- annotated_data/: contains generated clinical cases with gender annotations. All but corpus_manuel_annote.csv have been automatically annotated, by the gender detection system, and are in the subfolder "automatic annotations".
- bias_results/: contains csv files with gender metrics results and figures (in the fig/ subfolder) based on results obtained after bias metrics computation. Also contains analyse_phrase1.ipynb to look at the data and some other demographic features.
- generated_data/: contains the generated clinical cases in raw json files (in corresponding subfolder raw_json/) and their DataFrames versions
- ressources_lgq/: contains two json files with lists of epicene nouns as well as nouns referring to human entities
- src/: 
  - bias_exploration/:
    - correlation_bias-contraintes.py: a file to compute some various correlations on data
    - measure_bias.py: to compute Gender Gaps, Gender Shifts and some other stats
    - visualize_all.ipynb: notebook to generate some figures on the entire corpus
    - visualize_neutralprompts.ipynb: notebook to generate the same figures as _total but only on generations that answer a neutral prompt
  - preparation/:
    - evaluation.py: to evaluate the gender detection system (based on manual annotations)
    - gender_detection_adaptation: the gender detection system that has been adapted for P3 (and for clinical context) from [https://github.com/FannyDucel/GenderBiasCoverLetter]()
    - preprocess_files.py: to convert generations from json to csv (for raw json generations)
    - repetition.py: to compute repetitions rates in generations (to select better corpus)
    - classif_report/ : subfolder containing output txt file of evaluation.py and a csv file with the texts that have incorrect automatic annotations

# How to replicate the experiments
## Fine-tune your LLMs
TODO and TBA

## Generate your clinical cases
TODO and TBA

## Evaluate bias

1. preprocess_files.py
2. gender_detection_adaptation.py
3. evaluation.py (if manual annotations)
4. compute stats (measure_bias.py, ipynb for visualization + correlation_bias-contraintes et repetition)