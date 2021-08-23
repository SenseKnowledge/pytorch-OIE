# -*- coding: utf-8 -*-
import json


def fmt_label(tokens, label):
    span, rang = eval(label)

    # continuous
    assert len(rang) == rang[-1] - rang[0] + 1

    start = 0
    for i in range(rang[0]):
        start += len(tokens[i]) + 1  # space

    end = start
    for i in range(rang[0], rang[-1] + 1):
        end += len(tokens[i]) + 1
    end -= 1  # exclude last space

    return span, (start, end)


def fmt_oie2016(path):
    cache = {}

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')

            text, pred = line[:2]
            tokens = text.split(' ')

            if text not in cache:
                cache[text] = []

            _cache = {}
            pred, rang = fmt_label(tokens, pred)
            _cache['pred'] = (pred, rang)

            for i, argn in enumerate(line[2:]):
                argn, rang = fmt_label(tokens, argn)
                _cache[f'arg{i}'] = (argn, rang)

            cache[text].append(_cache)

    with open(path + '.json', 'w', encoding='utf-8') as f:

        for text, label in cache.items():
            f.write(json.dumps({'text': text, 'label': label}) + '\n')

    with open(path + '.fmt', 'w', encoding='utf-8') as f:
        for text, label in cache.items():
            for _label in label:
                args = sorted([t for t in _label if 'arg' in t])

                f.write('\t'.join([text, _label['pred'][0]] + [_label[argn][0] for argn in args]) + '\n')


if __name__ == '__main__':
    fmt_oie2016('train.oie')
    fmt_oie2016('test.oie')
    fmt_oie2016('dev.oie')
