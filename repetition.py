"""Use repetitiveness scores to compute averages and correlations between scores_reps (and/or has_reps, prop_avant_boucle, debuts_boucles)
and model/pathology/generated gender"""

import pandas as pd
import glob

#for file in glob.glob("generations_scores_repetitions/*.csv"):
for file in glob.glob("annotated_data/*trf*"):
    #model = file.split("/")[-1].split("_")[0]
    model = file.split("/")[-1].split("_")[1]
    df = pd.read_csv(file)
    df.fillna("N/A", inplace=True)

    #df_annotated = pd.read_csv(f"annotated_data/generations_{model}_10-consts_infos_gender_trf.csv")
    #df["Identified_gender"] = df_annotated["Identified_gender"]
    print(file)
    print(df["scores_reps"].mean(), df["scores_reps"].std())
    print(df.groupby("pathologie")["scores_reps"].mean())

    patho = {el:i  for i,el in enumerate(list(set(df['pathologie']))) }
    df['pathologie'] = df['pathologie'].replace(patho)
    print("Corrélation scores_reps x patho",df["scores_reps"].corr(df["pathologie"]))

    print(df.groupby("sex_prompt")["scores_reps"].mean())
    sex_prompt = {el: i for i, el in enumerate(list(set(df['sex_prompt'])))}
    df['sex_prompt'] = df['sex_prompt'].replace(sex_prompt)
    print("Corrélation scores_reps x sex_prompt", df["sex_prompt"].corr(df["scores_reps"]))

    print(df.groupby("Identified_gender")["scores_reps"].mean())
    gender = {el: i for i, el in enumerate(list(set(df['Identified_gender'])))}
    df['Identified_gender'] = df['Identified_gender'].replace(gender)
    print("Corrélation scores_reps x Identified_gender", df["Identified_gender"].corr(df["scores_reps"]))
    print("\n*********************")
