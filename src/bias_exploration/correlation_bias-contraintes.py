"""To compute correlations between biais and constraints respect rates.
Check 3 hypotheses:
    1) Les textes genrés au féminin respectent moins les contraintes.
        => comparer moyenne respect contraintes selon genre générations
            >> Moyennes très similaires peu importe le genre du prompt. Par contre, textes générés au féminin respectent toujours moins que ceux au masculin. Mais corrélations pas significatives (mais positives).
    2) Les textes portant sur des maladies stéréotypiquement fém respectent moins les contraintes
        => comparer moy. Gender Gap et moy respect contraintes par maladie et par groupe de maladies selon genre du stéréo)
            >> Ovaires et seins font toujours partie des maladies qui respectent le moins les contraintes. Corrélations négatives mais un peu élevées (-0.2, -0.4).
    3) Les textes contredisant le genre stéréotypique de la maladie respectent moins les contraintes
        => corrélations entre Gender Shift (0 ou 1) et respect des contraintes (par phrase entre 0 et 1)
            >> NO SIGNIFICANT CORRELATIONS. ALL NEGATIVE AND VERY LOW.
"""

#TODO: make a big merged file of everything (or find a way to compute on all data to get global stats)

import pandas as pd
import glob
from scipy.stats import pearsonr

def correlation(token_csv_path, coeff="pearson"):
    """Computes correlations between Gender Shift (0 or 1) and constraints respect rate.
    ALso need to choose the correlation coefficient: Pearson, Kendall or Spearman"""
    df = pd.read_csv(token_csv_path)
    model = token_csv_path.split("_")[-1].split(".")[0]
    df['sex_prompt'] = df['sex_prompt'].replace({'féminin': 'Fem', 'masculin': 'Masc', 'neutre': 'Neutral'})
    # Removing generations from neutral prompts as Gender Shift is irrelevant in this case
    df = df.loc[df["sex_prompt"] != "Neutral"]

    df['gender_shift'] = 0
    df.loc[df['sex_prompt'] != df['Identified_gender'] , "gender_shift"] = 1
    #df.loc[df['sex_prompt'] == "Neutral", "gender_shift"] = "N/A"

    #df.to_csv(output_file)
    # Compute average respect rate for texts with a positive GS
    avg_gs1 = df['respect_contraintes'].loc[df['gender_shift'] == 1].mean()
    # Compute average respect rate for texts with a negative GS
    avg_gs0 = df['respect_contraintes'].loc[df['gender_shift'] == 0].mean()

    print("Average respect rate for texts with a positive GS (=bias)", round(avg_gs1,2), "and with a negative GS", round(avg_gs0,2))
    print("Correlation between respect contraintes and GS:")
    print(pearsonr(df["respect_contraintes"], df["gender_shift"]))
    return round(df["respect_contraintes"].corr(df["gender_shift"], method=coeff),4)

def avg_respect_per_gender(token_csv_path):
    """Computes average respect of constraints per gender: generated + of prompt"""
    df = pd.read_csv(token_csv_path)
    avg_generation = df.groupby(["Identified_gender"])["respect_contraintes"].mean()
    avg_prompt = df.groupby(["sex_prompt"])["respect_contraintes"].mean()
    print("Respect contraintes STD",df.groupby(["Identified_gender"])["respect_contraintes"].std())

    df['Identified_gender'] = df['Identified_gender'].replace({'Fem':0, 'Masc':1, 'Neutre':3, 'Ambigu':4})
    print("Correlations respect contraintes x Identified gender",round(df["respect_contraintes"].corr(df["Identified_gender"]), 4))
    print("Respect contraintes avg total", round(df["respect_contraintes"].mean(),2))

    return avg_generation, avg_prompt

def avg_respect_per_patho(token_csv_path):
    df = pd.read_csv(token_csv_path)
    avg_patho = df.groupby(["pathologie"])["respect_contraintes"].mean()

    df['pathologie'] = df['pathologie'].replace({"COVID-19":1, "colon":2, "depression":3, "drepanocytose":4, "infarctus":5, "osteoporose":6, "ovaire":7, "prostate":8, "sein":9, "vessie":10})
    print("Correlations respect contraintes x Pathology",round(df["respect_contraintes"].corr(df["pathologie"]), 4))
    print(pearsonr(df["respect_contraintes"], df["pathologie"]))
    print("Respect contraintes avg total", round(df["respect_contraintes"].mean(),2))
    return avg_patho

def correl_nb_respect_contr(generation_path):
    """Check if there's a correlation between the respected rate and the nb of constraints (in generated_data file)
    => no significant correlations, between -0.06 and -0.33"""
    df = pd.read_csv(generation_path)
    print(pearsonr(df["nb_contraintes"], df["respect_contraintes"]))
    return df["nb_contraintes"].corr(df["respect_contraintes"])

def correl_respect_repet(generation_path):
    """Check if there's a correlation between the respected rate and the nb of constraints (in generated_data file)
    => no significant correlations, between -0.06 and -0.33"""
    df = pd.read_csv(generation_path)
    print(pearsonr(df["scores_reps"], df["respect_contraintes"]))
    return df["scores_reps"].corr(df["respect_contraintes"])


def correl_gender_patho(generation_path):
    """Check if there's a correlation between the respected rate and the nb of constraints (in generated_data file)
    => no significant correlations, between -0.06 and -0.33"""
    df = pd.read_csv(generation_path)
    df['Identified_gender'] = df['Identified_gender'].replace({'Fem':0, 'Masc':1, 'Neutre':3, 'Ambigu':4})
    df['pathologie'] = df['pathologie'].replace({"COVID-19":1, "colon":2, "depression":3, "drepanocytose":4, "infarctus":5, "osteoporose":6, "ovaire":7, "prostate":8, "sein":9, "vessie":10})

    print(pearsonr(df["Identified_gender"], df["pathologie"]))

    return df["Identified_gender"].corr(df["pathologie"])

def avg_repetition(generation_path):
    df = pd.read_csv(generation_path)
    return df["scores_reps"].mean()

"""
for file in glob.glob("../../annotated_data/*_trf.csv"):
    print(file)
    print(avg_repetition(file))
"""


full_corpus = "../../annotated_data/automatic_annotations/full_corpus.csv"
print(correlation(full_corpus))
print(avg_respect_per_gender(full_corpus))
print(avg_respect_per_patho(full_corpus))
print(correl_gender_patho(full_corpus))
print(correl_respect_repet(full_corpus))

exit()

df_list = []
for file in glob.glob("../../annotated_data/automatic_annotations/full*"):
    df = pd.read_csv(file)
    model = file.split("_")[-5].split(".")[0]
    print(model, round(df["respect_contraintes"].mean(), 3))
    # to merge all files into one big df
    ##
    df = pd.read_csv(file)
    df["model"] = model
    df_list.append(df)
    print(correlation(file), end="\n")
    print(avg_respect_per_gender(file), end="\n")
    print(avg_respect_per_patho(file), end="\n")
    print("*"*50)

#total_df = pd.concat([df for df in df_list])
#total_df.to_csv("annotated_data/generations_full-corpus.csv")


# Note : vigogne-7b and 13B respect the most the constraints.
#print(correlation())