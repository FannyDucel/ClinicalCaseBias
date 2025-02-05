import pandas as pd
import glob

# select 10 generations randomly per topic and create a csv with all these generations (10 topics*10 generation*6 models)
df_total = []
for llm_path in glob.glob("generations_*trf.csv"):
    df = pd.read_csv(llm_path)
    topics = list(set(df["pathologie"]))
    df_combined = df[df["pathologie"]==topics[0]].sample(5).reset_index()
    for topic in topics[1:]:
        df_new = df[df["pathologie"]==topic].sample(5).reset_index()
        df_combined = pd.concat([df_combined,df_new])
    df_combined['model'] = llm_path.split("_")[-3]
    df_combined['manual_gender'] = ''
    df_total.append(df_combined)

df_combined_total = pd.concat(df_total)
df_combined_total.drop('Unnamed: 0', axis=1, inplace=True)
df_combined_total.to_csv("corpus_manuel.csv")