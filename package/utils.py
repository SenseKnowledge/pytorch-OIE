# -*- coding: utf-8 -*-
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

def dump_oie_file(path: str, obj):
    pass