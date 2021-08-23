# -*- coding: utf-8 -*-
import torch

from typing import List, Dict
import json


def read_json_data(path: str) -> List[Dict]:
    """
    Read OIE format data
        sentence \t predicate \t arg0 \t arg1 \t arg2 \t arg3

    Parameters
    ----------
    path: str
        path to OIE format file

    Returns
    -------
    [
        {
         'text': ...
         'label': [
             {
                 'pred': ...
                 'arg0': ...
                  ...
             }
             ...
        ]
            ...
        }
        ...
    ]

    """
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def att_CE_Loss(logits, attention_mask, label):
    loss_fn = torch.nn.CrossEntropyLoss()
    logits = logits.view(-1, logits.size(-1))
    label = torch.where(attention_mask.view(-1) == 1, label.view(-1), torch.tensor(loss_fn.ignore_index).type_as(label))
    loss = loss_fn(logits, label)
    return loss
