import math
import transformers
from typing import Dict, List, Optional, Sequence
import torch
from vigogne.data_utils import IGNORE_INDEX


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int=None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        IGNORE_INDEX = -100

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # dtype = torch.long
        # input_ids, labels = tuple([torch.LongTensor(instance[key]) for instance in instances] for key in ("input_ids", "labels"))
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        if self.pad_to_multiple_of is not None:
            max_length_index, max_length = max(
                enumerate([len(input_ids_) for input_ids_ in input_ids]), key=lambda x: x[1]
            )
            # n_padding = ((max_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of - max_length
            n_padding = math.ceil(max_length / self.pad_to_multiple_of) * self.pad_to_multiple_of - max_length
            # Pad the longest example to pad_to_multiple_of * N
            input_ids[max_length_index].extend([self.pad_token_id] * n_padding)
            labels[max_length_index].extend([IGNORE_INDEX] * n_padding)

        input_ids = [torch.LongTensor(input_ids_) for input_ids_ in input_ids]
        labels = [torch.LongTensor(labels_) for labels_ in labels]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.pad_token_id),
        )
