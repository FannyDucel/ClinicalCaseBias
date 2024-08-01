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

# TODO : changer corpus var to arg
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
corpus = "all"

for file in glob.glob(f"annotated_data/*_trf.csv"):
    #print(file)
    df = pd.read_csv(file)
    modele = file.split('_')[2]
    df["modele"] = modele
    dic_df[modele] = df
#file = "annotated_data/generations_vigogne-2-7b_10-consts_infos_gender_trf.csv"
#df = pd.read_csv(file)

"""if corpus == "stereoFem":
    df = df[df["pathologie"].isin(["sein", "osteoporose", "ovaire", "depression"])]
if corpus == "stereoMasc":
    df = df[df["pathologie"].isin(["COVID-19", "prostate", "infarctus", "drepanocytose"])]
if corpus == "stereoNeutre":
    df = df[df["pathologie"].isin(["colon", "vessie"])]"""

#modele = file.split('_')[1]
#df["modele"] = modele
#dic_df[modele] = df

data_genre = pd.concat(list(dic_df.values()), ignore_index=True)
data_genre.replace({"Ambigu": "Ambiguous", "Fem": "Feminine", "Masc": "Masculine", "Neutre": "Neutral"}, inplace=True)
# todo : ajouter colonne avec maladie ? (chaque fichier de réf équivaut à une maladie)
#try:
topics = list(set(data_genre['pathologie']))
#except KeyError:
    #topics = list(set(data_genre['fichier_ref']))


def trier_dic(dic, reverse_=True):
    L = [[effectif, car] for car, effectif in dic.items()]
    L_sorted = sorted(L, reverse=reverse_)
    return [[car, effectif] for effectif, car in L_sorted]

def exploration_donnees_per_topic(dataset, topic):
    dataset = dataset[dataset["pathologie"] == topic]

    x_fig = dataset["Identified_gender"].value_counts(normalize=True)
    x = dataset["Identified_gender"].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
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
    df.replace({"masculin":"Masculine", "féminin":"Feminine"}, inplace=True)

    #df['gender_shift'] = np.where((df['sex_prompt'] != df['Identified_gender']) & (df['sex_prompt'] == "neutre") & (
                #df['Identified_gender'] != "Neutral"), 1, 0)

    # exclusion du neutre
    df = df[df.sex_prompt != "neutre"].copy()
    #df['gender_shift'] = np.where((df['sex_prompt'] != df['Identified_gender']) & (df['Identified_gender'] != "Neutral"), 1, 0)
    df["gender_shift"] = 0
    df.loc[(df['sex_prompt'] != df['Identified_gender']) & (df['Identified_gender'] != "Neutral"), "gender_shift"] = 1
    #print(df.shape)

    # print IDs of files that have a positive gender shift
    positive_gf = df['pathologie'][df["gender_shift"] == 1].to_list()
    #positive_gf = df.loc[df["gender_shift"] == 1, "pathologie"].to_list()
    print("List of pathologies with positive GS :", set(positive_gf))
    # GS per pathology: group by pathology and avg on the subcorpus
    print("Mean Gender Shift per pathology")
    print(df.groupby(['pathologie'])["gender_shift"].mean().nlargest(10))
    print("\nStandard deviation:", round(df['gender_shift'].std(),3))
    print("\nAvg GS per model, sorted:")
    print(df.groupby(['modele'])["gender_shift"].mean().sort_values())
    df.to_csv("bias_results/gender_shift.csv")
    return df['gender_shift'].mean()

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
    df.to_csv(f"bias_results/gender_gap_10_{corpus}.csv")

# neutre: with GG computed taking only into accounts generations from neutral prompts
#df_gendergap("neutre",modele)
#df_gendergap("all",modele)
#exit()

""" Computing results """

all_sorted_gap, all_masc_gap, all_fem_gap = gender_gap(topics,"all")
mean_gap_total = sum([el[1] for el in all_sorted_gap])/len(all_sorted_gap)
print("====== ON ALL PROMPTS (GENDERED + NEUTRAL)  ======")
print("The global Gender Gap is of", mean_gap_total)
print("The 10 diseases with the highest Gender Gaps are", all_sorted_gap[:10])
print("The 10 diseases with the lowest Gender Gaps are", all_sorted_gap[-10:])

all_sorted_gap, all_masc_gap, all_fem_gap = gender_gap(topics,"neutre")
mean_gap_total = sum([el[1] for el in all_sorted_gap])/len(all_sorted_gap)
print("\n====== ONLY ON NEUTRAL  ======")
print("The global Gender Gap is of", mean_gap_total)
print("The 10 diseases with the highest Gender Gaps are", all_sorted_gap[:10])
print("The 10 diseases with the lowest Gender Gaps are", all_sorted_gap[-10:])

print("\n")
print("\nThe global Gender Shift is of", gender_shift(data_genre))
# /!/ High GS for feminine but because it's the masculine prompts that are not respected (so a bit tricky),
# e.g. more difficult to make texts on men for stereotypically feminine diseases than the other way around
# Note: only 2000 rows for Gender Shift because exclude all neutral prompts
#Ovaire with 50% GS equals = only feminine generations so always respect when feminine prompts but never when masc