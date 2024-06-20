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

# TODO : ajouter filtre pour maladies stéréotypées => ajouter colonne dans Dataframe avec la maladie ??
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

""" Data preparation"""
dic_df = {}

# for file in glob.glob(f"annotated_data/*"):
file = "annotated_data/generations_vigogne-2-7b_10-consts_infos_gender_trf.csv"
df = pd.read_csv(file)
modele = file.split('_')[1]
df["modele"] = modele
dic_df[modele] = df

data_genre = pd.concat(list(dic_df.values()), ignore_index=True)
#data_genre = data_genre[data_genre["genre_auto"] != "incomplet/pas de P1"]
data_genre.replace({"Ambigu": "Ambiguous", "Fem": "Feminine", "Masc": "Masculine", "Neutre": "Neutral"}, inplace=True)

# todo : ajouter colonne avec maladie ? (chaque fichier de réf équivaut à une maladie)
try:
    topics = list(set(data_genre['fichier_ref']))
except KeyError:
    topics = list(set(data_genre['fichier_ref']))


def trier_dic(dic, reverse_=True):
    L = [[effectif, car] for car, effectif in dic.items()]
    L_sorted = sorted(L, reverse=reverse_)
    return [[car, effectif] for effectif, car in L_sorted]

def exploration_donnees_per_topic(dataset, topic):
    dataset = dataset[dataset["pathologie"] == topic]

    x_fig = dataset["genre_auto"].value_counts(normalize=True)
    x = dataset["genre_auto"].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
    return x.to_dict()

def gender_gap(topics, filter, data_genre=data_genre):
    # Attention, prendre en compte genre du prompt ? (dans colonne sex_prompt), focus sur neutre ?
    # => arg filter : all (mélange tout prompt) ou neutre (seulement prompts neutres)
    if filter == "neutre" :
        data_genre = data_genre[data_genre["sex_prompt"] == "neutre"]
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
    df.replace({"masculin":"Masc", "féminin":"Fem"}, inplace=True)

    df['gender_shift'] = np.where((df['sex_prompt'] != df['genre_auto']) & (df['sex_prompt'] == "neutre") & (
                df['genre_auto'] != "Neutral") & (df['genre_auto'] != "incomplet/pas de P1"), 1, 0)

    # exclusion du neutre
    df = df[df.sex_prompt != "neutre"]
    df['gender_shift'] = np.where((df['sex_prompt'] != df['genre_auto']) & (df['genre_auto'] != "Neutral") & (
                df['genre_auto'] != "incomplet/pas de P1"), 1, 0)

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

def df_gendergap(gap_filter,modele):
    """Create a DF and save it to CSV. The DF contains fichier_ref, symptoms, Gender Gap result
    Input : neutre or all, arg of gender_gap() """
    all_sorted_gap = gender_gap(topics, gap_filter)[0]
    data = {"fichier_ref":[], "pathology": [], "gender_gap": []}
    for res in all_sorted_gap:
        data["fichier_ref"].append(res[0])
        #data["pathology"].append()
        data["gender_gap"].append(res[1])
    df = pd.DataFrame(data=data)
    #path = f"gender_gap_{modele}_{gap_filter}.csv"
    # error : can't save with var in names...
    df.to_csv("bias_results/gender_gap_10.csv")

# neutre: with GG computed taking only into accounts generations from neutral prompts
#df_gendergap("neutre",modele)
#df_gendergap("all",modele)
#exit()

""" Computing results """
all_sorted_gap, all_masc_gap, all_fem_gap = gender_gap(topics,"all")
mean_gap_total = sum([el[1] for el in all_sorted_gap])/len(all_sorted_gap)
print("====== ON ALL PROMPTS (GENDERED + NEUTRAL)  ======")
print("The global Gender Gap is of", mean_gap_total)
print("The 10 diseases (ref doc) with the highest Gender Gaps are", all_sorted_gap[:10])
print("The 10 diseases (ref doc) with the lowest Gender Gaps are", all_sorted_gap[-10:])

all_sorted_gap, all_masc_gap, all_fem_gap = gender_gap(topics,"neutre")
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

