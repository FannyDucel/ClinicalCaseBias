import pandas as pd
import json
import glob

# convertir JSON forme liste de dicos [{"fichier":"file", "reference":"cas clinique ref", "constraints":"prompt", "candidats": ["generation",...],{dic2]
# - fichier : fichier réel, d'où provient le champ "référence" (référence pour faire BLEU, ROUGE)
# - input : l'ensemble des contraintes avec notamment âge et Sexe (seul truc que le modèle voit)
# - candidats : les 5 générations pour l'exemple précis
# - "constraints" c'est pour ensuite rechercher facilement si elles sont présentes dans le texte

def ouvrir_json(chemin):
    with open(chemin, encoding="utf-8") as f:
        toto = json.load(f)
    return toto

def json_to_df(chemin_json):
    """Converts a json file with a list of dicts (from Nicolas' generations) to a Dataframe with the file ID (fichier_ref),
    the generation, the prompted sex and the prompted age (when they're present)"""
    contenu_df = []
    modele = chemin_json.split("/")[-1].split(".")[0]
    for dic in ouvrir_json(chemin_json) :
        # sex and age are in the "input" value, in a string formatted like "Sexe : féminin ; âge : 2 ; ..."
        # But sex is not always determined (neutral setting)
        #sex = dic["input"].split(";")[0].split(":")[-1].strip() if "Sexe" in dic["input"].split(";")[0] else "Undetermined"
        #age = dic["input"].split(";")[1].split(":")[-1].strip() if "Age" in dic["input"].split(";")[1] else dic["input"].split(";")[0].split(":")[-1].strip()
        for i, generation in enumerate(dic["candidats"]):
            texte = {"fichier_ref": dic["fichier"], "pathologie":dic["pathologie"], "generation": generation, "input":dic["input"], "sex_prompt":dic["sexe"], "age_prompt":dic["age"], "nb_contraintes":dic["nb_contraintes"], "respect_contraintes":dic["respected_rate"][i]}
            contenu_df.append(texte)
    df = pd.DataFrame(contenu_df)
    df.to_csv(f"generated_data/generations_{modele}.csv")

#vigogne = "generated_data/raw_json/vigogne-2-7b_10-consts_infos.json"
#json_to_df(vigogne)

for file in glob.glob(f"generated_data/raw_json/data_biais/*"):
    json_to_df(file)
