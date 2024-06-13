""" ADAPTATION ONGOING
TO RUN with command such as : python measure_bias.py FR neutral
If ran for the gendered setting, returns both the Gender Gap and the Gender Shift.
If ran for neutral, returns the Gender Gap."""
import json

import pandas as pd
from tabulate import tabulate
import sys
import glob
import argparse
import sys
import numpy as np

# TODO : ajouter filtre pour maladies stéréotypées
# class MyParser(argparse.ArgumentParser):
#     def error(self, message):
#         sys.stderr.write('error: %s\n' % message)
#         self.print_help()
#         sys.exit(2)

# parser = MyParser()
# # parser.add_argument('language', choices=['FR', 'IT'])
# parser.add_argument('experiment_type', choices=['neutral', 'gendered'])
#
# args = parser.parse_args()
#
# #language = sys.argv[1]
# type_expe = sys.argv[1]
#language = "FR"
#type_expe = "neutral"
dic_df = {}

for file in glob.glob(f"annotated_data/*"):
    df = pd.read_csv(file)
    modele = file.split('_')[1]
    df["modele"] = modele
    dic_df[modele] = df

data_genre = pd.concat(list(dic_df.values()), ignore_index=True)
#data_genre = data_genre[data_genre["Identified_gender"] != "incomplet/pas de P1"]
data_genre.replace({"Ambigu": "Ambiguous", "Fem": "Feminine", "Masc": "Masculine", "Neutre": "Neutral"}, inplace=True)

"""Calculer l'Écart Genré selon les modèles"""
def trier_dic(dic, reverse_=True):
    L = [[effectif, car] for car, effectif in dic.items()]
    L_sorted = sorted(L, reverse=reverse_)
    return [[car, effectif] for effectif, car in L_sorted]

# todo : ajouter colonne avec maladie ? (chaque fichier de réf équivaut à une maladie)
try:
    topics = list(set(data_genre['fichier_ref']))
except KeyError:
    topics = list(set(data_genre['fichier_ref']))

def exploration_donnees_per_topic(dataset, topic):
    dataset = dataset[dataset["fichier_ref"] == topic]

    x_fig = dataset["Identified_gender"].value_counts(normalize=True)
    x = dataset["Identified_gender"].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
    return x.to_dict()

def gender_gap(topics, filter, data_genre=data_genre):
    # Attention, prendre en compte genre du prompt ? (dans colonne sex_prompt), focus sur Undetermined ?
    # => arg filter : all (mélange tout prompt) ou undetermined (seulement prompts neutres)
    if filter == "undetermined" :
        data_genre = data_genre[data_genre["sex_prompt"] == "Undetermined"]
    gap = {}  # seulement topic et gap
    for topic in topics:
        op = exploration_donnees_per_topic(data_genre, topic)
        # gap masc-fem donc si positifs, biaisé vers Masc, si négatif, biaisé vers Fem
        try:
            m = float(op['Masculine'][:-1])
        except KeyError:
            m = 0

        try:
            f = float(op['Feminine'][:-1])
        except KeyError:
            f = 0

        gap[topic] = m - f
        sorted_gap = trier_dic(gap)

        masc_gap = [el for el in sorted_gap if el[1] > 0]

        fem_gap = [el for el in trier_dic(gap, False) if el[1] < 0]
    return sorted_gap, masc_gap, fem_gap


def gender_shift(df):
    """Renvoie la probabilité que le prompt ne soit pas respecté (= nb de fois où le texte est généré dans le genre opposé ou ambigu)"""
    df.replace({"masculin":"Masculine", "féminin":"Feminine"}, inplace=True)

    df['gender_shift'] = np.where((df['sex_prompt'] != df['Identified_gender']) & (df['sex_prompt'] == "Undetermined") & (
                df['Identified_gender'] != "Neutral") & (df['Identified_gender'] != "incomplet/pas de P1"), 1, 0)

    # exclusion du neutre
    df = df[df.sex_prompt != "Undetermined"]
    df['gender_shift'] = np.where((df['sex_prompt'] != df['Identified_gender']) & (df['Identified_gender'] != "Neutral") & (
                df['Identified_gender'] != "incomplet/pas de P1"), 1, 0)

    # print IDs of files that have a positive gender shift
    positive_gf = df.loc[df["gender_shift"] == 1, "fichier_ref"].to_list()
    print("List of files with positive GS :", [(el, id_symptoms(el)) for el in positive_gf])
    # df.to_csv("gender_shift_noneutral.csv")
    return sum(df['gender_shift']) / len(df['gender_shift'])

def id_symptoms(id) :
    """For a given document reference (id), returns the DISO elements of the json file, e.g. the symptoms associated with
    the clinical cases. To use in order to link IDs of the highest Gender Gaps to the symptoms."""
    with open("generated_data/raw_json/vigogne-2-7b_5-consts.json", "r") as f:
        data = json.load(f)
    for dic in data :
        if dic["fichier"] == id :
            symptoms = [el[0] for el in dic["constraints"] if el[-1] == "DISO"]
            return symptoms


all_sorted_gap, all_masc_gap, all_fem_gap = gender_gap(topics,"all")
mean_gap_total = sum([el[1] for el in all_sorted_gap])/len(all_sorted_gap)
print("====== ON ALL PROMPTS (GENDERED + NEUTRAL)  ======")
print("The global Gender Gap is of", mean_gap_total)
print("The 10 diseases (ref doc) with the highest Gender Gaps are", all_sorted_gap[:10])
print("The 10 diseases (ref doc) with the lowest Gender Gaps are", all_sorted_gap[-10:])

all_sorted_gap, all_masc_gap, all_fem_gap = gender_gap(topics,"undetermined")
mean_gap_total = sum([el[1] for el in all_sorted_gap])/len(all_sorted_gap)
print("\n====== ONLY ON NEUTRAL  ======")
print("The global Gender Gap is of", mean_gap_total)
print("\n\tTop 10 highest Gender Gaps (id, symptoms list)")
for id in all_sorted_gap[:10] :
    print(id, id_symptoms(id[0]))
print("\n\tTop 10 lowest Gender Gaps (id, symptoms list)")
for id in all_sorted_gap[-10:] :
    print(id, id_symptoms(id[0]))

print("\n")
print("\nThe global Gender Shift is of", gender_shift(data_genre))

