# -*- coding: utf-8 -*-
import torch
import numpy as np

pre_tag2idx = {
    'O': 0,
    'P-B': 1, 'P-I': 2,
}

arg_tag2idx = {
    'O': 0,
    'A0-B': 1, 'A0-I': 2,
    'A1-B': 3, 'A1-I': 4,
    'A2-B': 5, 'A2-I': 6,
    'A3-B': 7, 'A3-I': 8,
}

NUM_OF_ARGS = 4


def decode_confidence_score(prob_pre, prob_arg, infobox):
    """
    Decode confidence score from logits and infobox

    Parameters
    ----------
    prob_pre
    prob_arg
    infobox: List[Dict], [{'P':[], 'P_ids':[], 'A0':[], 'A0_ids':[], ...}]

    Returns
    -------

    """
    confidence = []
    for _prob_arg, _infobox in zip(prob_arg, infobox):

        _confidence = 0.

        # predicate
        _confidence += max(prob_pre[_infobox['pred'][1][0]]).item()

        # argument
        for arg_n in range(NUM_OF_ARGS):
            if len(_infobox[f'arg{arg_n}'][1]) == 0:
                continue

            ids = _find_begin(_infobox[f'arg{arg_n}'][1])
            _confidence += np.mean([max(_prob_arg[i]).item() for i in ids])

        confidence.append(_confidence)

    return confidence


def _find_begin(ids):
    begins = [ids[0]]
    for i in range(1, len(ids)):
        if ids[i] - ids[i - 1] != 1:
            begins.append(ids[i])
    return begins


def convert_tags_to_infobox(pre_tags, arg_tags, tokens):
    """
    Convert tags to infobox

    Parameters
    ----------
    pre_tags
    arg_tags
    tokens

    Returns
    -------

    """
    infobox = []
    for cur_pre_tags, cur_arg_tags in zip(pre_tags, arg_tags):
        cur_result = {}

        # predicate
        span_pre_ids = [i for i, tag in enumerate(cur_pre_tags) if tag != pre_tag2idx['O']]
        span_pre = []
        if span_pre_ids != 0:
            for i, token in enumerate(tokens):
                if i in span_pre_ids:
                    span_pre.append(token)
        else:
            # must have the predicate
            continue

        cur_result['pred'] = (span_pre, span_pre_ids)

        # argument
        for arg_n in range(NUM_OF_ARGS):
            span_arg_ids = [i for i, tag in enumerate(cur_arg_tags)
                            if tag in {arg_tag2idx[f'A{arg_n}-B'], arg_tag2idx[f'A{arg_n}-I']}]

            span_arg = []
            if span_arg_ids != 0:
                for i, token in enumerate(tokens):
                    if i in span_arg_ids:
                        span_arg.append(token)

            cur_result[f'arg{arg_n}'] = (span_arg, span_arg_ids)

        infobox.append(cur_result)
    return infobox


def convert_pre_bio_to_mask(tensor):
    """
    converting predicate index with 'O' tag to 0 and other indexes are converted to 1.

    Parameters
    ----------
    tensor: predicate tagged tensor with the shape of (B, L), where B is the batch size, L is the sequence length.

    Returns
    -------
    masked binary tensor with the same shape.
    """

    res = tensor.clone().detach()
    res[tensor != pre_tag2idx['O']] = 1
    res[tensor == pre_tag2idx['O']] = 0
    return res.bool()


def convert_pre_tag_tensor_to_ids(pre_tags):
    """
    convert predicate tag to its indexes

    Parameters
    ----------
    pre_tags

    Returns
    -------

    """
    return [[idx.item() for idx in (pred_tag != pre_tag2idx['O']).nonzero()] for pred_tag in pre_tags]


def filter_pre_tags(pre_tags, input_ids, tokenizer):
    """
    Filter useless tokens by converting them into 'Outside' tag.
    We treat 'Inside' tag before 'Beginning' tag as meaningful signal,
    so changed them to 'Beginning' tag unlike [Stanovsky et al., 2018].

    Parameters
    ----------
    pre_tags: predicate tags with the shape of (B, L).
    input_ids: list format sentence pieces with the shape of (B, L)
    tokenizer: transformers.Tokenizer

    Returns
    -------
    tensor of filtered predicate tags with the same shape.
    """
    assert pre_tags.size() == input_ids.size()

    # filter by tokens ([CLS], [SEP], [PAD] tokens should be allocated as 'O')
    for pred_idx, cur_tokens in enumerate(input_ids):
        for tag_idx, token in enumerate(cur_tokens):
            if token in tokenizer.convert_tokens_to_ids(
                    [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]):
                pre_tags[pred_idx][tag_idx] = pre_tag2idx['O']

    # filter by tags
    pred_copied = pre_tags.clone()
    for pred_idx, cur_pred_tag in enumerate(pred_copied):
        flag = False
        tag_copied = cur_pred_tag.clone()
        for tag_idx, tag in enumerate(tag_copied):
            if not flag and tag == pre_tag2idx['P-B']:
                flag = True
            elif not flag and tag == pre_tag2idx['P-I']:
                pre_tags[pred_idx][tag_idx] = pre_tag2idx['P-B']
                flag = True
            elif flag and tag == pre_tag2idx['O']:
                flag = False
    return pre_tags


def filter_arg_tags(arg_tags, pre_tags, input_ids, tokenizer):
    """
    Same as the description of @filter_pred_tags().

    Parameters
    ----------
    arg_tags: argument tags with the shape of (B, L).
    pre_tags: predicate tags with the same shape.
        It is used to force predicate position to be allocated the 'Outside' tag.
    input_ids: list of string tokens with the length of L.
        It is used to force special tokens like [CLS] to be allocated the 'Outside' tag.
    tokenizer: transformers.Tokenizer

    Returns
    -------
    tensor of filtered argument tags with the same shape.
    """

    # filter by tokens ([CLS], [SEP], [PAD] tokens should be allocated as 'O')
    for arg_idx, (cur_arg_tag, _input_ids) in enumerate(zip(arg_tags, input_ids)):
        for tag_idx, token in enumerate(_input_ids):
            if token in tokenizer.convert_tokens_to_ids(
                    [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]):
                arg_tags[arg_idx][tag_idx] = arg_tag2idx['O']

    # filter by tags
    arg_copied = arg_tags.clone()
    for arg_idx, (cur_arg_tag, cur_pred_tag) in enumerate(zip(arg_copied, pre_tags)):
        pred_ids = [idx[0].item() for idx in (cur_pred_tag != pre_tag2idx['O']).nonzero()]
        arg_tags[arg_idx][pred_ids] = arg_tag2idx['O']
        cur_arg_copied = arg_tags[arg_idx].clone()
        flag_idx = 999
        for tag_idx, tag in enumerate(cur_arg_copied):
            if tag == arg_tag2idx['O']:
                flag_idx = 999
                continue
            arg_n = (tag - 1) // 2  # 0: A0 / 1: A1 / ...
            inside = (tag - 1) % 2  # 0: begin / 1: inside
            if not inside and flag_idx != arg_n:
                flag_idx = arg_n
            # connect_args
            elif not inside and flag_idx == arg_n:
                arg_tags[arg_idx][tag_idx] = arg_tag2idx[f'A{arg_n}-I']
            elif inside and flag_idx != arg_n:
                arg_tags[arg_idx][tag_idx] = arg_tag2idx[f'A{arg_n}-B']
                flag_idx = arg_n
    return arg_tags


def divide_pre_tag(pre_tags):
    """
    Divide each single batch based on predicted predicates.
    It is necessary for predicting argument tags with specific predicate.

    Parameters
    ----------
    pre_tags: tensor of predicate tags with the shape of (B, L)

    Returns
    -------
        list of tensors with the shape of (B, P, L), the number P can be different for each batch.

    Examples
    -------
        >> tensor([[2, 0, 0, 1, 0, 1, 0, 2, 2, 2],
                   [2, 2, 2, 0, 1, 0, 1, 2, 2, 2],
                   [2, 2, 2, 2, 2, 2, 2, 2, 0, 1]])

        >> [tensor([[2., 0., 2., 2., 2., 2., 2., 2., 2., 2.],
                    [2., 2., 0., 1., 2., 2., 2., 2., 2., 2.],
                    [2., 2., 2., 2., 0., 1., 2., 2., 2., 2.],
                    [2., 2., 2., 2., 2., 2., 0., 2., 2., 2.]]),
            tensor([[2., 2., 2., 0., 1., 2., 2., 2., 2., 2.],
                    [2., 2., 2., 2., 2., 0., 1., 2., 2., 2.]]),
            tensor([[2., 2., 2., 2., 2., 2., 2., 2., 0., 1.]])]


    """
    total_pre_tags = []
    for cur_pred_tag in pre_tags:
        cur_sent_pred = []
        begin_ids = [idx[0].item() for idx in (cur_pred_tag == pre_tag2idx['P-B']).nonzero()]
        for i, b_idx in enumerate(begin_ids):
            cur_pred = [pre_tag2idx['O']] * cur_pred_tag.shape[0]
            cur_pred[b_idx] = pre_tag2idx['P-B']
            if i == len(begin_ids) - 1:
                end_idx = cur_pred_tag.shape[0]
            else:
                end_idx = begin_ids[i + 1]
            for j, tag in enumerate(cur_pred_tag[b_idx:end_idx]):
                if tag.item() == pre_tag2idx['O']:
                    break
                elif tag.item() == pre_tag2idx['P-I']:
                    cur_pred[b_idx + j] = pre_tag2idx['P-I']
            cur_sent_pred.append(cur_pred)
        total_pre_tags.append(cur_sent_pred)
    return [torch.Tensor(_pre_tags) for _pre_tags in total_pre_tags]
