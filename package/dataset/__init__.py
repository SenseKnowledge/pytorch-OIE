# -*- coding: utf-8 -*-
import json

from typing import List, Dict


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


def dump_json_file(path: str, obj: List[Dict]):
    with open(path, 'w', encoding='utf-8') as f:
        for _obj in obj:
            text = _obj['text']

            for _label in _obj['label']:
                msg = text + '\t' + _label['score'] + '\t' + _label['pred']

                args = sorted([argn for argn in _label if argn.startswith('arg')])

                for arg in args:
                    msg += '\t' + _label[arg][0]

                f.write(msg + '\n')
