# -*- coding: utf-8 -*-
import torch

from package.utils import dense, att_CE_Loss
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Union


class AlphaConfig:
    """
    LMs + MLP + CLS Config
    """

    def __init__(self,
                 pretrained_model_name_or_path, pos_embedding_dim,
                 fc_1_hidden_size, fc_1_dropout_rate, fc_2_hidden_size, fc_2_dropout_rate,
                 pre_tag_size, arg_tag_size):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.pos_embedding_dim = pos_embedding_dim

        self.pre_tag_size = pre_tag_size
        self.arg_tag_size = arg_tag_size

        # predicate layer
        self.fc_1_hidden_size = fc_1_hidden_size
        self.fc_1_dropout_rate = fc_1_dropout_rate

        # argument layer
        self.fc_2_hidden_size = fc_2_hidden_size
        self.fc_2_dropout_rate = fc_2_dropout_rate


class AlphaModel(torch.nn.Module):
    """
    LMs + MLP + CLS Model

    ref: Multi^2OIE: Multilingual Open Information Extraction Based on Multi-Head Attention with BERT. AAAI. 2020
    """

    def __init__(self, config: AlphaConfig):
        super(AlphaModel, self).__init__()

        # Language model
        self.lm = AutoModel.from_pretrained(config.pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path)
        _config = self.lm.config

        # Position embedding
        self.POS = torch.nn.Embedding(3, config.pos_embedding_dim, padding_idx=0)
        self.pre_tag_size = config.pre_tag_size
        self.arg_tag_size = config.arg_tag_size

        # Predicate layer
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(_config.hidden_size, config.fc_1_hidden_size),
            torch.nn.Dropout(config.fc_1_dropout_rate),
            torch.nn.ReLU(),
            torch.nn.Linear(config.fc_1_hidden_size, self.pre_tag_size, bias=False)
        )

        # Argument layer
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(_config.hidden_size * 2 + config.pos_embedding_dim, config.fc_2_hidden_size),
            torch.nn.Dropout(config.fc_2_dropout_rate),
            torch.nn.ReLU(),
            torch.nn.Linear(config.fc_2_hidden_size, config.fc_2_hidden_size),
            torch.nn.Dropout(config.fc_2_dropout_rate),
            torch.nn.ReLU(),
            torch.nn.Linear(config.fc_2_hidden_size, self.arg_tag_size, bias=False)
        )

    def state_dict(self):
        """
        Exclude the weights of LMs
        """
        state_dict = super(AlphaModel, self).state_dict()
        return {k: v for k, v in state_dict.items() if not k.startswith('lm')}

    def _predict_predicate(self, hidden):
        x = self.fc1(hidden)
        return x

    def _predict_argument(self, hidden, hidden_pre, hidden_pos):
        x = torch.cat([hidden, hidden_pre, hidden_pos], dim=-1)
        x = self.fc2(x)
        return x

    @property
    def _exclude_token_ids(self):
        return self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token])

    @torch.no_grad()
    def forward(self, text: Union[str, List[str]]) -> List[Dict]:
        """
        OIE Pipeline

        Parameters
        ----------
        text: Union[str, list[str]]
            input text

        Returns
        -------
        List[Dict]
        """
        device = next(self.parameters()).device

        token = self.tokenizer(text, padding=True, return_offsets_mapping=True)
        input_ids = torch.LongTensor(token.input_ids).to(device)
        attention_mask = torch.ByteTensor(token.attention_mask).to(device)
        offset_mapping = token.offset_mapping
        hidden = self.lm(input_ids, attention_mask)[0]

        hidden_pre = self._predict_predicate(hidden)
        pre_tags = torch.argmax(hidden_pre, dim=-1)
        pre_prob = torch.softmax(hidden_pre, dim=-1)

        pre_tags = dense.filter_pre_tags(pre_tags, input_ids, self._exclude_token_ids)
        pre_tags = dense.divide_pre_tags(pre_tags)

        cache = []
        B = len(pre_tags)
        for i in range(B):
            P = pre_tags[i].size(0)

            if P:
                mask = dense.convert_pre_tags_to_mask(pre_tags[i])
                hidden_ = hidden[i:i + 1].repeat(P, 1, 1)
                input_ids_ = input_ids[i:i + 1].repeat(P, 1)
                pre_pos_ids = dense.get_pre_location_ids(mask, input_ids_)
                pre_pos_hidden = self.POS(pre_pos_ids)
                hidden_pre_ = dense.get_pre_hidden(mask, hidden_)
                hidden_arg = self._predict_argument(hidden_, hidden_pre_, pre_pos_hidden)

                arg_prob = torch.softmax(hidden_arg, dim=-1)
                arg_tags = torch.argmax(hidden_arg, dim=-1)
                arg_tags = dense.filter_arg_tags(arg_tags, pre_tags[i], input_ids_, self._exclude_token_ids)

                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                _cache = dense.convert_tags_to_infobox(pre_tags[i], pre_prob[i], arg_tags, arg_prob, tokens)
                cache.append(_cache)
            else:
                cache.append([])

        cache = dense.postprocess(text, cache, offset_mapping, self.tokenizer)
        return cache

    def loss(self, input_ids, attention_mask, pre_label_all, pre_label, arg_label):

        with torch.no_grad():
            hidden = self.lm(input_ids, attention_mask)[0]

        hidden_pre = self._predict_predicate(hidden)
        loss_pre = att_CE_Loss(hidden_pre, attention_mask, pre_label_all)

        pre_mask = dense.convert_pre_tags_to_mask(pre_label)
        pre_pos_ids = dense.get_pre_location_ids(pre_mask, input_ids)
        hidden_pre_pos = self.POS(pre_pos_ids)
        hidden_pre = dense.get_pre_hidden(pre_mask, hidden)
        hidden_arg = self._predict_argument(hidden, hidden_pre, hidden_pre_pos)
        loss_arg = att_CE_Loss(hidden_arg, attention_mask, arg_label)

        return loss_pre, loss_arg
