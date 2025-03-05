import glob
import json
import torch
import json


def get_constraints(path):
    with open(path, "r", encoding="utf-8") as fin:
        contenu = json.load(fin)
    #constraints = [(a["instruction"], a["input"], a["output"], a["fichier"], a["constraints"]) for a in contenu]
    constraints = [(a["instruction"], a["input"], a["output"], a["fichier"]) for a in contenu]

    return constraints

def get_train_cas(path):
    path_train = path  # "../../corpus/CAS/train"
    textes = []
    for fic in glob.glob(f"{path_train}/*.txt"):
        with open(fic, "r", encoding="utf-8") as fin:
            contenu = fin.read()
            textes.append(contenu)
    # print(len(textes))
    # print(textes[50])
    textes = [f"<|startoftext|> {d[0:-1]} <|endoftext|>" for d in textes]
    # print(textes[50])
    return textes


def get_train_e3c(path):
    path_train = path  # "../../corpus/E3C-French/splits/split_1"
    train = f"{path_train}/train"
    val = f"{path_train}/val"
    textes = []
    for fic in glob.glob(f"{train}/*.txt"):
        with open(fic, "r", encoding="utf-8") as fin:
            contenu = fin.read()
            textes.append(contenu)
    for fic in glob.glob(f"{val}/*.txt"):
        with open(fic, "r", encoding="utf-8") as fin:
            contenu = fin.read()
            textes.append(contenu)
    # print(len(textes))
    # print(textes[0])
    textes = [f"<|startoftext|> {d[0:]} <|endoftext|>" for d in textes]
    # print(textes[0])
    return textes


def lire_e3c(path):
    chemin = path  # "../../corpus/E3C-French/fichiers_json"
    contenu = []
    total_tokens = 0
    for fic in glob.glob("%s/*" % chemin):
        with open(fic, "r", encoding="utf-8") as fin:
            dic = json.load(fin)
            if dic["type"] in ["journal", "pubmed"]:
                nb_toks = len(dic["text"].split())
                total_tokens += nb_toks
                contenu.append(dic["text"])
    # print(contenu[0])
    # print(total_tokens)
    contenu = [f"<|startoftext|> {d[0:-1]} <|endoftext|>" for d in contenu]  # a changer selon version e3c (espace au
    # début de chaque doc ou non)
    return contenu


def lire_e3c_annotations(path):
    chemin = path  # "../corpus/E3C-French/fichiers_xml"
    contenu = []
    for fic in glob.glob("%s/*" % chemin):
        with open(fic, "r", encoding="utf-8") as fin:
            contenu.append(fin.read())
    contenu = [f"<|startoftext|> {d[0:-1]} <|endoftext|>" for d in contenu]
    return contenu


def data_collator(features):
    first = features[0]
    batch = {}
    for k, v in first.items():
        batch[k] = torch.tensor([f[k] for f in features])
    return batch


def load_special_tokens(bos_and_eos: bool, path_special_tokens=None):
    special_tokens = []
    if path_special_tokens is not None:
        with open(path_special_tokens, "r", encoding="utf-8") as fin:
            special_tokens = json.load(fin)
    if bos_and_eos:
        special_tokens += ["<|startoftext|>", "<|endoftext|>"]
    return special_tokens

def read_json(path):
    dic = {}
    with open(path, "r", encoding="utf-8") as fin:
        dic = json.load(fin)
    return dic
