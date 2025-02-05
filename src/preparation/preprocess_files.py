"""
Convert JSON files containing lists of dictionaries [{"fichier":"file", "reference":"cas clinique ref", "constraints":"prompt", "candidats": ["generation",...],{dic2]
- fichier : real file, from where the field "reference" comes from (to compute BLEU, ROUGE)
- input : set of constraints, notably age and sex (only thing that the model sees)
- candidates : the 5 generated texts
- constraints : to easily see if they are present in the text
"""

import pandas as pd
import json
import glob

def ouvrir_json(chemin):
    with open(chemin, encoding="utf-8") as f:
        toto = json.load(f)
    return toto

def cas_multi(delimitations_cas):
    """Returns 1 if the text contains multiple cases"""
    multi = 0
    if len(delimitations_cas) > 1:
        multi = 1
    return multi

def json_to_df(chemin_json):
    """
    Converts a json file with a list of dicts (from clinical cases generations) to a Dataframe with the file ID (fichier_ref),
    the generation, the prompted sex and the prompted age (when they're present)
    """
    contenu_df = []
    modele = chemin_json.split("/")[-1].split("_")[0]
    for dic in ouvrir_json(chemin_json) :
        # sex and age are in the "input" value, in a string formatted like "Sexe : féminin ; âge : 2 ; ..."
        # But sex is not always determined (neutral setting)
        #sex = dic["input"].split(";")[0].split(":")[-1].strip() if "Sexe" in dic["input"].split(";")[0] else "Undetermined"
        #age = dic["input"].split(";")[1].split(":")[-1].strip() if "Age" in dic["input"].split(";")[1] else dic["input"].split(";")[0].split(":")[-1].strip()
        for i, generation in enumerate(dic["candidats"]):
            texte = {"fichier_ref": dic["fichier"], "pathologie":dic["pathologie"], "generation": generation, "input":dic["input"], "sex_prompt":dic["sexe"], "age_prompt":dic["age"], "nb_contraintes":dic["nb_contraintes"], "respect_contraintes":dic["respected_rate"][i],
                     "scores_reps":dic["scores_reps"][i], "delimitations_cas":len(dic["delimitations_cas"][i]), "cas_multiples":cas_multi(dic["delimitations_cas"][i]),
                     "new_gen":dic["is_new"][i]}
            contenu_df.append(texte)
    df = pd.DataFrame(contenu_df)
    df.to_csv(f"../../filtered_generations/generations_{modele}.csv")



for file in glob.glob(f"../../filtered_generations/*.json"):
    json_to_df(file)
