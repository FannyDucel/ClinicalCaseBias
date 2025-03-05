from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from utils import *
from functools import partial
from collator import DataCollatorForSupervisedDataset
from typing import Dict, List, Optional, Sequence
from vigogne.utils.packing import Concatenator, ModerateConcatenator
from processors import data_processor


class DataProcess:
    def __init__(self, tokenizer_name: str, block_size: int) -> object:
        self.block_size = block_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        #self.tokenizer = load_tokenizer(tokenizer_name)

    def tokenize_function_lm(self, examples):
        return self.tokenizer(examples["text"])

    def group_texts(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // self.block_size) * self.block_size
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def add_special_tokens_to_tokenizer(self, special_tokens_dict):
        self.tokenizer.add_special_tokens(special_tokens_dict)
        # self.tokenizer.bos_token = "<|startoftext|>"
        # self.tokenizer.eos_token = "<|endoftext|>"

    def get_len_dataset(self, tokenized_datasets):
        # nb_tokens = 0
        longueurs = []
        for el in tokenized_datasets:
            for ids in tokenized_datasets[el]["input_ids"]:
                # nb_tokens += len(ids)
                longueurs.append(len(ids))
        print(f"Longueur moyenne : {np.mean(longueurs)} tokens")
        print(f"Ecart type : {np.std(longueurs)}")
        print(f"Maximum : {np.max(longueurs)} | Minimum : {np.min(longueurs)}")
        return np.sum(longueurs)

    # data: list of text
    #def prepare_data_lm(self, data: [], batch_size, special_tokens=None):
    def prepare_data_lm(self, train_file: str, val_file: str, batch_size: int, special_tokens=None):

        #train, test = train_test_split(data, test_size=0.20)
        train_data = read_json(train_file)
        val_data = read_json(val_file)

        #train = [x["train_example"] for x in train_data]
        #val = [x["train_example"] for x in val_data]

        train = [x["reference"] for x in train_data]
        val = [x["reference"] for x in val_data]

        train = [f"<|startoftext|> {x}</s>" for x in train]
        val = [f"<|startoftext|> {x}</s>" for x in val]

        #print(train[0])

        #print(val[0])

        print(len(train))
        print(len(val))

        datasets = DatasetDict()
        df_train = pd.DataFrame(train, columns=["text"])
        df_val = pd.DataFrame(val, columns=["text"])
        datasets["train"] = Dataset.from_pandas(df_train)
        datasets["eval"] = Dataset.from_pandas(df_val)

        #datasets["train"] = load_dataset("json", data_files=train_file)["train"]
        #datasets["eval"] = load_dataset("json", data_files=val_file)["train"]

        print(len(self.tokenizer))

        if special_tokens is not None:
            self.add_special_tokens_to_tokenizer(special_tokens)

        tokenized_datasets = datasets.map(
            self.tokenize_function_lm,
            batched=True,
            num_proc=4,
            remove_columns=["text"],
            #remove_columns=next(iter(datasets.values())).column_names,
            desc="process dataset"
        )

        #print(tokenized_datasets)

        #print(tokenized_datasets["train"][0])

        #lm_datasets = tokenized_datasets.map(self.group_texts, batched=True, batch_size=1000, num_proc=4)

        lm_datasets = tokenized_datasets.map(
            #ModerateConcatenator(block_size=self.block_size),
            Concatenator(block_size=self.block_size),
            batched=True,
            desc=f"packing texts in blocks of {self.block_size}"
        )

        print(lm_datasets)

        #print(lm_datasets["train"][0])

        #print(self.tokenizer.decode(lm_datasets["train"][0]["input_ids"]))

        ## Si pas avec le trainer d'huggingface
        #train_loader = DataLoader(
        #    lm_datasets["train"], batch_size=batch_size, num_workers=1, collate_fn=data_collator
        #)
        #val_loader = DataLoader(
        #    lm_datasets["validation"], batch_size=batch_size, num_workers=1, collate_fn=data_collator
        #)

        return lm_datasets["train"], lm_datasets["eval"], self.tokenizer
        #return train_loader, val_loader, self.tokenizer

    def prepare_data_instruct(self, train_file: str, val_file: str, template: str, batch_size: int, special_tokens=None, hf_trainer=False):
        datasets = DatasetDict()
        datasets["train"] = load_dataset("json", data_files=train_file)["train"]
        datasets["eval"] = load_dataset("json", data_files=val_file)["train"]

        print(len(datasets["train"]))
        print(len(datasets["eval"]))

        if special_tokens is not None:
            self.add_special_tokens_to_tokenizer(special_tokens)

        special_tokens_dict = dict()
        #special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        special_tokens_dict["pad_token"] = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens(special_tokens_dict)

        #process_function = SUPPORTED_PROCESSOR_TEMPLATES["instruct"].process_example
        #processor = SUPPORTED_PROCESSORS.get("alpaca")

        #process_function_p = partial(
        #    process_function,
        #    tokenizer=self.tokenizer,
        #)

        processor = data_processor()

        tokenized_datasets = datasets.map(
            processor.process_example,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "mask_input": True,
                "template": template
            },
            #batched=True,
            num_proc=4,
            remove_columns=next(iter(datasets.values())).column_names,
            desc="process dataset"
        )

        print(tokenized_datasets)
        print(self.tokenizer.decode(tokenized_datasets["eval"][0]["input_ids"]))

        lm_datasets = tokenized_datasets.map(
            #ModerateConcatenator(block_size=256),
            Concatenator(block_size=self.block_size),
            batched=True,
            desc=f"packing texts in blocks of {self.block_size}"
        )

        #print(lm_datasets)
        #print(self.tokenizer.decode(lm_datasets["eval"][0]["input_ids"]))

        if hf_trainer:
            return lm_datasets["train"], lm_datasets["eval"], self.tokenizer

        else:

            collator = DataCollatorForSupervisedDataset(self.tokenizer.pad_token_id, 8)

            train_loader = DataLoader(
                lm_datasets["train"],
                #tokenized_datasets["train"],
                batch_size=batch_size,
                num_workers=1,
                collate_fn=collator
            )
            val_loader = DataLoader(
                lm_datasets["eval"],
                #tokenized_datasets["eval"],
                batch_size=batch_size,
                num_workers=1,
                collate_fn=collator
            )

            return train_loader, val_loader, self.tokenizer
