"""Adaptation of the gender detection system for *third* person singular in French. Based on morpho-syntactic gender markers
and leveraging semantic information."""

import json
import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm
import glob


nlp = spacy.load("fr_dep_news_trf")

def get_gender(text, details=False):
    """
    Apply linguistic rules based on Spacy tags to detect the first person singular gender
    markers in a text.

    Args:
        text (str): The text to be analyzed (= for which we want to find the author's gender).
        details (bool): (False by default), True to get the details (token, lemma, pos, dep, gender, number) of all tokens that are detected as gender markers, False otherwise.

    Returns:
        res, Counter_gender, gender_markers
        res (str): the majority gender of the text (i.e. the annotated gender of the author of the text)
        Counter_gender (Counter): the details of the numbers of markers found per gender
        gender_markers (list): the list of identified gender markers
    """

    text = text.replace("  ", " ")
    doc = nlp(text)

    # list of gender-neutral (épicène) job titles from DELA, with Profession:fs:ms, to check and filter out if they're identified as Masc when used without a masc DET
    with open("../../ressources_lgq/epicenes_corr.json", encoding="utf-8") as f:
        epicene_jobs = json.load(f)
        epicene_jobs.append("tout")
        epicene_jobs.append("toute")

    with open("../../ressources_lgq/ressource_p3.json", encoding="utf-8") as f:
        agents_hum = json.load(f)

    # Remove medical or judiciary job titles that are often present but to refer to a third person, not the patient
    agents_hum = [el for el in agents_hum if el not in ["magistrat", "requérant", "requérante","magistrate", "toxicologue", "médecin", "docteur",
                                                        "docteure", "cardiologue", "gérontolongue", "gastroentérologue" ,"neurologue", "pneumologue", "dermatologue", "mycologue", "virologue", "immunologue", "bactériologue",
                                                        "podologue", "gynécologue", "radiologue", "allergologue"]]

    # list of the gender tags identified in the adj/verbs of the text
    gender = []
    # list of the tokens that have a gender tag (and are adj/verbs)
    gender_markers = []
    # pour présence de prénom/initiale(s) comme premier mot du texte => à réessayer avec condition que ce soit .head()
    prenom_initiale = []
    for sent in doc.sents:
        if "sexe féminin" in sent.text:
            gender.append("Fem")
            gender_markers.append("sexe féminin")
        if "sexe masculin" in sent.text:
            gender.append("Masc")
            gender_markers.append("sexe masculin")

        this_sent = []
        split_sent = str(sent).replace("'", ' ').split()
        for token in sent:
            this_sent.append(token.text.lower() + "-" + token.dep_)

            # 1. The token is a noun referring to a human agent or initials (names)
            cond_agt = token.text.lower() in agents_hum and token.pos_=="NOUN"
            if len(this_sent) == 1 and ((token.pos_ == "PROPN" and "nsubj" in token.dep_) or (token.text.isupper() and len(token)==2 or len(token)==4 and "." in token.text)):
                prenom_initiale.append(token.text)

            cond_agt_avt = [s for s in this_sent if "nsubj" in s] and [s for s in this_sent if "nsubj" in s][-1].split("-")[0] in agents_hum

            # 2. The token is an adj or past participle (that has an auxiliary different from "avoir") that refers to a human agent/initial
            cond_pos = (token.pos_ == "ADJ" or token.pos_ == "VERB")
            cond_noavoir = (("a-aux:tense" not in this_sent and "avoir-aux:tense" not in this_sent) or ("a-aux:tense" in this_sent and "été-aux:pass" in this_sent))
            cond_adj_pp = cond_pos and (
                    ((token.head.text.lower() in agents_hum or (
                                prenom_initiale and token.head.text == prenom_initiale[0])) and cond_noavoir) or (
                            token.head.pos_ != "NOUN" and cond_noavoir and cond_agt_avt))

            # Manually fix Spacy mistakes (mislabeling some Feminine words as Masculine ones)
            erreurs_genre = ["inscrite", "technicienne"]

            if cond_agt or cond_adj_pp:
                token_gender = token.morph.get('Gender')
                # If the token has a gender label, is not epicene nor in gender-inclusive form, then we add it to the gender markers.
                if token_gender and token.text.lower() not in epicene_jobs and "(" not in token.text.lower() and token.text.lower() not in erreurs_genre: #(e
                    gender.append(token_gender[0])
                    gender_markers.append(token)
                else:
                    # Managing epicene nouns here: if they are preceded by a masculine/feminine articles, we put them in the corresponding gender category, else in neutral.
                    if (token.text.lower() in epicene_jobs and len(this_sent)>1 and this_sent[-2] in ["un-det", "le-det"]) or token.text.lower()=="chef" and "chef" not in [str(tok) for tok in gender_markers]:
                        gender.append("Masc")
                        gender_markers.append(token)
                    if (token.text.lower() in epicene_jobs and len(this_sent)>1 and this_sent[-2] in ["une-det", "la-det"]) or token.text.lower() in erreurs_genre:  # or token.text=="Femme":
                        gender.append("Fem")
                        gender_markers.append(token)
                    if "(" in token.text.lower():
                        gender.append("Neutre")
                        gender_markers.append(token)

            if details:
                print(token.text.lower(), token.pos_, token.dep_, token.lemma_, token.morph.get("Gender"), token.morph.get("Number"))

    Counter_gender = Counter(gender)
    if len(Counter_gender) > 0:
        # The final result (= the gender of the token) is the majority gender, i.e. the gender that has the most markers in this text.
        res = Counter_gender.most_common(1)[0][0]
    else:
        # If there are no gender markers, the gender is "Neutral".
        res = "Neutre"

    counter_val = Counter_gender.values()
    if len(counter_val) > 1 and len(set(counter_val))==1:
        # If there are as many masculine as feminine markers, the category is "Ambiguous".
        res = "Ambigu"

    return res, Counter_gender, gender_markers


def detecter_genre(csv_path):
    """
    Apply gender detection system (from function get_gender) on the generations contained in a CSV file and append
    the results (manual annotations) in a new CSV file.

    Args:
        csv_path: A string -> the path of the CSV file containing the generated cover letters.
        This CSV file must have a column "output" (with the generated texts), a column "prompt" and "Theme" (pro. field).

    Returns:
        Nothing, creates a new annotated CSV file by appending the manual annotations
        (= new columns "Identified_gender" with the detected gender, "Detailed_counter" with the nb of markers found
        for each gender, and "Detailed_markers" with the list of identified gender markers and their associated gender)
    """

    df_lm = pd.read_csv(csv_path)

    lm = df_lm["generation"]
    lm.fillna("", inplace=True)

    total_gender = [] #list with all identified gender in order to add to pd
    total_counter = [] #list with all counters to get dic with identified genders details for each letter
    total_markers = [] #same for gendered words that led to this gender identification

    for lettre in tqdm(lm):
        gender = get_gender(lettre)
        total_gender.append(gender[0])
        total_counter.append(gender[1])
        total_markers.append(gender[2])

    df_lm["Identified_gender"]=total_gender
    df_lm["Detailed_counter"] = total_counter
    df_lm["Detailed_markers"] = total_markers

    #path = csv_path.split("/")[1]
    path = csv_path.split(".")[0].split("/")[-1]

    df_lm.to_csv("annotated_data/"+path+f"_gender_trf.csv")

for file in glob.glob(f"generated_data/*"):
    if "_infos.csv" in file:
        print(file)
        detecter_genre(file)
