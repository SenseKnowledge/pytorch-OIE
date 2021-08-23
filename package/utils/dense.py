# -*- coding: utf-8 -*-
import torch

pre_tag2idx = {
    'O': 0,
    'P-B': 1, 'P-I': 2,
}

# related to @filter_arg_tags
arg_tag2idx = {
    'O': 0,
    'A0-B': 1, 'A0-I': 2,
    'A1-B': 3, 'A1-I': 4,
    'A2-B': 5, 'A2-I': 6,
    'A3-B': 7, 'A3-I': 8,
}

NUM_OF_ARGS = 4


def filter_pre_tags(tags, input_ids, exclude_token_ids):
    """
    1. Filter useless tokens [CLS, SEP, PAD] by converting them into 'Outside' tag.

    Parameters
    ----------
    tags: Tensor
        predicate tags with the shape of (B, L).
    input_ids: Tensor
        sentence pieces with the shape of (B, L)
    exclude_token_ids: List[str]
        token ids shouldn't included

    Returns
    -------
    tensor of filtered predicate tags with the same shape.
    """

    # filter by tokens ([CLS], [SEP], [PAD] tokens should be allocated as 'O' in Bert)
    for i, cur_ids in enumerate(input_ids):
        for j, idx in enumerate(cur_ids):
            if idx in exclude_token_ids:
                tags[i][j] = pre_tag2idx['O']

    # filter by tags
    tags_copied = tags.clone()
    for i, cur_pred_tag in enumerate(tags):
        flag = False
        tag_copied = cur_pred_tag.clone()
        for j, tag in enumerate(tag_copied):
            if not flag and tag == pre_tag2idx['P-B']:
                flag = True
            elif not flag and tag == pre_tag2idx['P-I']:
                tags_copied[i][j] = pre_tag2idx['P-B']
                flag = True
            elif flag and tag == pre_tag2idx['O']:
                # swap flag
                flag = False

    return tags_copied


def filter_arg_tags(arg_tags, pre_tags, input_ids, exclude_token_ids):
    """
    Same as the description of @filter_pre_tags

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
    for i, (cur_arg_tag, cur_ids) in enumerate(zip(arg_tags, input_ids)):
        for j, idx in enumerate(cur_ids):
            if idx in exclude_token_ids:
                arg_tags[i][j] = arg_tag2idx['O']

    # filter by tags
    arg_copied = arg_tags.clone()
    for i, (cur_arg_tag, cur_pred_tag) in enumerate(zip(arg_tags, pre_tags)):
        pred_ids = [idx[0].item() for idx in (cur_pred_tag != pre_tag2idx['O']).nonzero()]
        arg_copied[i][pred_ids] = arg_tag2idx['O']
        cur_arg_copied = arg_copied[i].clone()
        flag_idx = 999

        # [HACK] related to tag2idx
        for j, tag in enumerate(cur_arg_copied):
            if tag == arg_tag2idx['O']:
                flag_idx = 999
                continue
            arg_n = (tag - 1) // 2  # 0: A0 / 1: A1 / ...
            inside = (tag - 1) % 2  # 0: begin / 1: inside
            if not inside and flag_idx != arg_n:
                flag_idx = arg_n
            # connect_args
            elif not inside and flag_idx == arg_n:
                arg_copied[i][j] = arg_tag2idx[f'A{arg_n}-I']
            elif inside and flag_idx != arg_n:
                arg_copied[i][j] = arg_tag2idx[f'A{arg_n}-B']
                flag_idx = arg_n
    return arg_copied


def divide_pre_tags(tags):
    """
    Divide each single batch based on predicted predicates.
    It is necessary for predicting argument tags with specific predicate.

    Parameters
    ----------
    tags: tensor of predicate tags with the shape of (B, L)

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
    cache = []
    for cur_pred_tag in tags:
        _cache = []
        begins = [idx[0].item() for idx in (cur_pred_tag == pre_tag2idx['P-B']).nonzero()]
        for i, begin in enumerate(begins):
            cur_pred = [pre_tag2idx['O']] * cur_pred_tag.shape[0]
            cur_pred[begin] = pre_tag2idx['P-B']
            if i == len(begins) - 1:
                end = cur_pred_tag.shape[0]
            else:
                end = begins[i + 1]
            for j, tag in enumerate(cur_pred_tag[begin:end]):
                if tag.item() == pre_tag2idx['O']:
                    break
                elif tag.item() == pre_tag2idx['P-I']:
                    cur_pred[begin + j] = pre_tag2idx['P-I']
            _cache.append(cur_pred)
        cache.append(torch.Tensor(_cache))
    return cache


def convert_pre_tags_to_mask(tensor):
    """
    Converting predicate index with 'O' tag to 0 and other indexes are converted to 1.

    Parameters
    ----------
    tensor: predicate tagged tensor with the shape of (B, L), where B is the batch size, L is the sequence length.

    Returns
    -------
    masked binary tensor with the same shape.
    """

    mask = torch.zeros_like(tensor)
    mask[tensor != pre_tag2idx['O']] = 1
    return mask.bool()


def get_pre_location_ids(mask, input_ids):
    """
    Get the location of predicate from its mask

    Parameters
    ----------
    mask:
    input_ids

    Returns
    -------
    Tensor, 1 for predicate tag and 0 for others
    """
    position_ids = torch.zeros_like(mask, dtype=torch.int)
    for i, (cur_mask, cur_ids) in enumerate(zip(mask, input_ids)):
        position_ids[i, :] += 2
        cur_nonzero = cur_mask.nonzero()
        start = torch.min(cur_nonzero).item()
        end = torch.max(cur_nonzero).item()
        position_ids[i, start:end + 1] = 1
        pad_start = max(cur_ids.nonzero()).item() + 1
        position_ids[i, pad_start:] = 0
    return position_ids


def get_pre_hidden(mask, hidden):
    """
    Get the predicate hidden from mask and hidden

    Parameters
    ----------
    mask: Tensor
    hidden: Tensor

    Returns
    -------

    """
    _, L, _ = hidden.shape
    hidden_pre = torch.zeros_like(hidden)
    for i, (cur_mask, cur_hidden) in enumerate(zip(mask, hidden)):
        pred_position = cur_mask.nonzero().flatten()
        pred_feature = torch.mean(cur_hidden[pred_position], dim=0, keepdim=True).repeat(L, 1)
        hidden_pre[i, :, :] = pred_feature
    return hidden_pre


def convert_tags_to_infobox(pre_tags, pre_prob, arg_tags, arg_prob, tokens):
    """
    Convert tags to infobox
    [{'pred': ([token], [ids]), 'arg0': ...} ...]

    Parameters
    ----------
    pre_tags
    pre_prob
    arg_tags
    arg_prob
    tokens

    Returns
    -------
    InfoBox
    """
    cache = []
    for cur_pre_tags, cur_arg_tags, cur_arg_prob in zip(pre_tags, arg_tags, arg_prob):
        _cache = {'score': 0.}

        # predicate
        span_pre = []
        span_pre_ids = [i for i, tag in enumerate(cur_pre_tags) if tag != pre_tag2idx['O']]

        if span_pre_ids:
            for i, token in enumerate(tokens):
                if i in span_pre_ids:
                    span_pre.append(token)
        else:
            # must have the predicate
            continue

        _cache['pred'] = (span_pre, span_pre_ids)
        _cache['score'] += max(pre_prob[span_pre_ids[0]]).item()

        # argument
        for arg_n in range(NUM_OF_ARGS):
            span_arg_ids = [i for i, tag in enumerate(cur_arg_tags)
                            if tag.item() in {arg_tag2idx[f'A{arg_n}-B'], arg_tag2idx[f'A{arg_n}-I']}]

            prob = [max(cur_arg_prob[i]).item() for i in span_arg_ids if i == arg_tag2idx[f'A{arg_n}-B']]
            if prob:
                prob = sum(prob) / len(prob)
            else:
                prob = 0.5
            _cache['score'] += prob

            span_arg = []
            if span_arg_ids:
                for i, token in enumerate(tokens):
                    if i in span_arg_ids:
                        span_arg.append(token)

            _cache[f'arg{arg_n}'] = (span_arg, span_arg_ids)

        _cache['score'] /= 5.

        cache.append(_cache)
    return cache


def postprocess(text, infobox, offset_mapping, tokenizer):
    """
    Post-process Result

    Parameters
    ----------
    text
    infobox
    offset_mapping
    tokenizer

    Returns
    -------

    """

    def _diff(a):
        return [(_1 - _2) == 1 for _1, _2 in zip(a[1:], a[:-1])] if a else [False]

    cache = []
    for _text, _infobox, _offset_mapping in zip(text, infobox, offset_mapping):
        _cache = {'text': _text, 'label': []}
        for __infobox in _infobox:
            __cache = {}
            for tag, box in __infobox.items():
                if tag == 'score':
                    __cache['score'] = box
                    continue

                token, ids = box

                if all(_diff(ids)):
                    token = tokenizer.convert_tokens_to_string(token).replace(' ', '')
                    start, end = ids[0], ids[-1]
                    start, end = _offset_mapping[start][0], _offset_mapping[end][-1]

                    __cache[tag] = (token, (start, end))
            _cache['label'].append(__cache)
        cache.append(_cache)
    return cache
