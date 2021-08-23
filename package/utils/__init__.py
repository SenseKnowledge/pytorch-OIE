# -*- coding: utf-8 -*-
import torch


def att_CE_Loss(logits, attention_mask, label):
    loss_fn = torch.nn.CrossEntropyLoss()
    logits = logits.view(-1, logits.size(-1))
    label = torch.where(attention_mask.view(-1) == 1, label.view(-1), torch.tensor(loss_fn.ignore_index).type_as(label))
    loss = loss_fn(logits, label)
    return loss
