"""Évaluer la détection automatique de genre :
1. Compter dans les annotations manuelles (pour voir ce que donnent les vrais résultats)
2. Comparer/évaluer la détection auto vs manuel
3. Essayer d'améliorer les scores de la détection auto : ajout de filtres ou essai de classif"""
from datetime import datetime
from typing import Dict, List
import re
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate
from sklearn.metrics import classification_report

def multiple_replace(string, rep_dict):
    pattern = re.compile("|".join([re.escape(k) for k in sorted(rep_dict,key=len,reverse=True)]), flags=re.DOTALL)
    return pattern.sub(lambda x: rep_dict[x.group(0)], string)

def common_prefix(list_of_strings: List[str]) -> str:
    # Start with first character of first string and keep going
    if len(list_of_strings) == 1:
        return list_of_strings[0]

    prefix = ''
    for i in range(len(list_of_strings[0])):
        new_prefix = list_of_strings[0][:i]
        cond = [string.startswith(new_prefix) for string in list_of_strings]
        if all(cond):
            prefix = new_prefix
        else:
            return prefix

    return prefix


def common_suffix(list_of_strings: List[str]) -> str:
    list_of_strings = [x[::-1] for x in list_of_strings]
    return common_prefix(list_of_strings)[::-1]

def prec_recall_fscore(file,corpus):
    """Use sklearn to get classification report and overall precision, recall and fscore"""
    df = pd.read_csv(file)

    #print("Ajout masque pour textes avec problèmes d'annotation manuelle")
    # FR100350/FR101419/FR101436/FR100795/FR100731 (à discuter, "nourrisson" puis "patient"),
    # FR100760 (pluriel mais décrit 2 hommes), FR101566 (pluriel), FR101084 (pluriel mais fém)
    # FR101356/FR101335/FR101585/FR100967/FR101193/FR101039/FR101709 (enfant), FR100319 (enfant mais fém),
    # FR101263 = 2 cas en 1 et 1 homme et 1 femme
    # FR100678 = nourrisson de sexe féminin mais "il" après
    if "e3c" in file:
        # + ajout ceux du fichiers_hs.csv de Nicolas
        mask = df['id'].isin(["FR100350","FR100678","FR101084 ","FR101419","FR101436","FR100795","FR100731", "FR100760", "FR101566", "FR101356","FR101335","FR101585","FR100967","FR101193","FR101039","FR101709",
                              "FR100201","FR100760","FR100924","FR100933","FR100940","FR100941","FR100951","FR100962","FR100964","FR100966","FR100976","FR100984","FR100997","FR101000","FR101052","FR101078","FR101084","FR101089","FR101093","FR101169","FR101329","FR101418","FR101431","FR101491","FR101564","FR101566","FR101584","FR101655","FR101727","FR101728","FR101736","FR101737","FR101741","FR101743","FR101745","FR101749","FR101754","FR101763","FR101766"])
        df = df[~mask]

    if "cas" in file :
        # cas de pluriels ou discutables
        mask = df['id'].isin(
            ["filepdf-855-cas", "filepdf-554-2-cas", "filepdf-533-6-cas", "filepdf-534-8-cas", "filepdf-554-3-cas", "filepdf-508-2-cas"
             , "filepdf-508-1-cas", "filepdf-23-cas"])
        df = df[~mask]
    #errors = df.query('genre_manuel != genre_auto')
    errors = df.query('genre_manuel != Identified_gender')

    n_annote = df.genre_manuel.count()
    errors.to_csv(f"../../annotated_data/errors_detection_{n_annote}_{corpus}.csv")

    y_true = df["genre_manuel"].loc[:n_annote].to_numpy()
    y_pred = df["Identified_gender"].loc[:n_annote].to_numpy()

    prec, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='macro')

    with open(f"classification_report_{n_annote}_{corpus}.txt", "w") as f:
        print(datetime.now(), file=f)
        print(file, file=f)
        print(classification_report(y_true, y_pred, digits=4), file=f) #target_names=labels,

    return prec, recall, fscore, support

print(prec_recall_fscore("../../annotated_data/corpus_manuel_annote.csv","full_annote"))
#print(prec_recall_fscore("test_121cas_gender_trf.csv"))
#print(prec_recall_fscore("preliminary_tests/test_500cas_gender_trf.csv", "cas"))
#print(prec_recall_fscore("preliminary_tests/test_300e3c_gender_trf.csv", "e3c"))
#print(prec_recall_fscore("preliminary_tests/generations_bloomz_gender_trf.csv", "bloomz"))
#print(prec_recall_fscore("preliminary_tests/generations_vigogne_gender_trf.csv", "vigogne"))
