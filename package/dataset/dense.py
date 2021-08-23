# -*- coding: utf-8 -*-
import torch

from package.utils import read_json_data
from package.utils.dense import pre_tag2idx, arg_tag2idx
from torch.utils.data import Dataset as _Dataset, DataLoader
from transformers import PreTrainedTokenizer
from collections import Counter
from random import randint


class Dataset(_Dataset):

    def __init__(self, data_path, tokenizer: PreTrainedTokenizer):
        self.data = read_json_data(data_path)
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        _data = self.data[item]
        text = _data['text']
        label = _data['label']

        # fill slots
        tags_pre, tags_arg = [], []
        for _label in label:

            if 'arg0' not in _label:
                _label['arg0'] = ''

            if 'arg1' not in _label:
                _label['arg1'] = ''

            if 'arg2' not in _label:
                _label['arg2'] = ''

            if 'arg3' not in _label:
                _label['arg3'] = ''

            pre, arg0, arg1, arg2, arg3 = _label['pred'], _label['arg0'], _label['arg1'], _label['arg2'], _label['arg3']
            _tags_pre, _tags_arg = self._convert_label_to_tags(text, pre, arg0, arg1, arg2, arg3)
            tags_pre.append(_tags_pre)
            tags_arg.append(_tags_arg)

        return text, tags_pre, tags_arg

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _convert_label_to_tags(text, pre, *arg):
        tags_pre = [pre_tag2idx['O']] * len(text)
        tags_arg = [arg_tag2idx['O']] * len(text)

        pre, (start, end) = pre
        tags_pre[start] = pre_tag2idx['P-B']
        tags_pre[start + 1:end] = [pre_tag2idx['P-I']] * (end - start - 1)

        for i, arg_n in enumerate(arg):
            if arg_n:
                arg_n, (start, end) = arg_n

                tags_arg[start] = arg_tag2idx[f'A{i}-B']
                tags_arg[start + 1:end] = [arg_tag2idx[f'A{i}-I']] * (end - start - 1)

        return tags_pre, tags_arg

    def collate_fn(self, batch):
        """collate_fn for 'torch.utils.data.DataLoader'
        """

        # flatten
        batch_text = []
        batch_tags_pre = []
        batch_tags_arg = []
        for text, tags_pre, tags_arg in batch:
            batch_text.append(text)
            batch_tags_pre.append(tags_pre)
            batch_tags_arg.append(tags_arg)

        token = self.tokenizer(batch_text, return_offsets_mapping=True)
        batch_tags_pre_all = [self._combine_pre_tags(_) for _ in batch_tags_pre]

        # sample args pair for each predicate
        batch_tags_pre_sample, batch_tags_arg_sample = [], []
        for tags_pre, tags_arg in zip(batch_tags_pre, batch_tags_arg):
            i = randint(0, len(tags_pre) - 1)
            batch_tags_pre_sample.append(tags_pre[i])
            batch_tags_arg_sample.append(tags_arg[i])

        # align the label
        # Bert mat split a word 'AA' into 'A' and '##A'
        batch_tags_pre_all = [self._align_label(offset, tags, pre_tag2idx) for offset, tags in
                              zip(token['offset_mapping'], batch_tags_pre_all)]
        batch_tags_pre = [self._align_label(offset, tags, pre_tag2idx) for offset, tags in
                          zip(token['offset_mapping'], batch_tags_pre_sample)]
        batch_tags_arg = [self._align_label(offset, tags, arg_tag2idx) for offset, tags in
                          zip(token['offset_mapping'], batch_tags_arg_sample)]

        token = self.tokenizer.pad(token)
        input_ids = torch.LongTensor(token.input_ids)
        attention_mask = torch.ByteTensor(token.attention_mask)
        max_len = input_ids.size(-1)
        return (input_ids, attention_mask,
                self._pad_label(batch_tags_pre_all, max_len, pre_tag2idx),
                self._pad_label(batch_tags_pre, max_len, pre_tag2idx),
                self._pad_label(batch_tags_arg, max_len, arg_tag2idx))

    @staticmethod
    def _combine_pre_tags(tags):

        tag_combine = tags[0].copy()
        for tag in tags[1:]:

            for i in range(len(tag_combine)):

                if tag[i] == pre_tag2idx['P-B'] or (
                        tag[i] == pre_tag2idx['P-I'] and tag_combine[i] != pre_tag2idx['P-B']):
                    tag_combine[i] = tag[i]
        return tag_combine

    @staticmethod
    def _align_label(offset, tags, tag2idx):

        def _tag_vote(tags):
            return Counter(tags).most_common(1)[0][0]

        label_align = []
        for i, (start, end) in enumerate(offset):

            if start == end:
                label_align.append(tag2idx['O'])
            else:
                label_align.append(_tag_vote(tags[start:end]))
        return label_align

    @staticmethod
    def _pad_label(labels, max_len, tag2idx):
        labels = [(label + [tag2idx['O']] * (max_len - len(label))) for label in labels]
        return torch.LongTensor(labels)
