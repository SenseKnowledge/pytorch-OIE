# -*- coding: utf-8 -*-
import torch

from package.seq_utils import pre_tag2idx, arg_tag2idx, filter_pre_tags, filter_arg_tags, divide_pre_tag
from package.seq_utils import convert_pre_bio_to_mask, convert_tags_to_infobox, decode_confidence_score
from transformers import AutoModel, AutoTokenizer
from typing import Union, List


class BertSeqTagConfig:

    def __init__(self, pos_embedding_dim,
                 fc_1_hidden_size, fc_1_dropout_rate, fc_2_hidden_size, fc_2_dropout_rate,
                 pretrained_model_name_or_path):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        self.pos_embedding_dim = pos_embedding_dim
        self.fc_1_hidden_size = fc_1_hidden_size
        self.fc_1_dropout_rate = fc_1_dropout_rate
        self.fc_2_hidden_size = fc_2_hidden_size
        self.fc_2_dropout_rate = fc_2_dropout_rate


class BertSeqTagModel(torch.nn.Module):
    """Bert + Seq"""

    def __init__(self, config: BertSeqTagConfig):
        super(BertSeqTagModel, self).__init__()

        # Language Model
        self.lm = AutoModel.from_pretrained(config.pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path)
        _config = self.lm.config

        # Position embedding
        self.POS = torch.nn.Embedding(3, config.pos_embedding_dim, padding_idx=0)
        self.pre_tag_size = len(pre_tag2idx)
        self.arg_tag_size = len(arg_tag2idx)

        # Predicate
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(_config.hidden_size, config.fc_1_hidden_size),
            torch.nn.Dropout(config.fc_1_dropout_rate),
            torch.nn.ReLU(),
            torch.nn.Linear(config.fc_1_hidden_size, self.pre_tag_size, bias=False)
        )

        # Argument
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(_config.hidden_size * 2 + config.pos_embedding_dim, config.fc_2_hidden_size),
            torch.nn.Dropout(config.fc_2_dropout_rate),
            torch.nn.ReLU(),
            torch.nn.Linear(config.fc_2_hidden_size, self.arg_tag_size, bias=False)
        )

    def state_dict(self):
        """
        Exclude the weights of LMs

        """
        state_dict = super(BertSeqTagModel, self).state_dict()
        return {k: v for k, v in state_dict.items() if not k.startswith('lm')}

    @torch.no_grad()
    def forward(self, text: Union[str, List[str]]):
        """ OIE pipeline

        Parameters
        ----------
        text: str, list of str

        Returns
        -------
        InforBox
        [{'pred': ..., 'arg0': ..., 'arg1': ..., ...}
         ...
        ]
        """
        token = self.tokenizer(text, padding=True, return_offsets_mapping=True, return_tensors='pt')
        hidden = self.lm(token.input_ids, token.attention_mask)[0]

        hidden_pre = self._predict_predicate(hidden)
        pre_tags = torch.argmax(hidden_pre, dim=-1)
        pre_prob = torch.softmax(hidden_pre, dim=-1)
        pre_tags = filter_pre_tags(pre_tags, token.input_ids, self.tokenizer)
        pre_tags_sent = divide_pre_tag(pre_tags)

        result = []
        for i, _pre_tags in enumerate(pre_tags_sent):
            n_pre = _pre_tags.size(0)
            if n_pre != 0:
                _pre_prob = pre_prob[i]

                _pre_mask = convert_pre_bio_to_mask(_pre_tags)
                _hidden = hidden[i:i + 1].repeat(n_pre, 1, 1)
                _input_ids = token.input_ids[i:i + 1].repeat(n_pre, 1)

                _pre_pos_ids = self._get_position_ids(_pre_mask, _input_ids)
                _hidden_pre_pos = self.POS(_pre_pos_ids)
                _hidden_pre = self._get_pre_hidden(_pre_mask, _hidden)
                _hidden_arg = self._predict_argument(_hidden, _hidden_pre, _hidden_pre_pos)

                _arg_prob = torch.softmax(_hidden_arg, dim=-1)
                _arg_tags = torch.argmax(_hidden_arg, dim=-1)
                _arg_tags = filter_arg_tags(_arg_tags, _pre_tags, _input_ids, self.tokenizer)

                infobox = convert_tags_to_infobox(_pre_tags, _arg_tags,
                                                  self.tokenizer.convert_ids_to_tokens(token.input_ids[i]))
                confidence = decode_confidence_score(_pre_prob, _arg_prob, infobox)

                for j, score in enumerate(confidence):
                    infobox[j]['score'] = score

                result.append(infobox)
            else:
                result.append([])


        # offset_mapping = token.offset_mapping.cpu().numpy()
        # result = self._refine_infobox(text, result, offset_mapping)

        return result

    def _refine_infobox(self, text, infobox, offset_mapping):

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

                    if not all(_diff(ids)):
                        __cache[tag] = ('', tuple())
                    else:
                        token = self.tokenizer.convert_tokens_to_string(token)
                        start, end = ids[0], ids[-1]
                        start, end = _offset_mapping[start][0], _offset_mapping[end][-1] + 1

                        __cache[tag] = (token, (start, end))
                _cache['label'].append(__cache)
            cache.append(_cache)
        return cache

    def _predict_predicate(self, hidden):
        x = self.fc1(hidden)
        return x

    def _predict_argument(self, hidden, hidden_pre, hidden_pos):
        x = torch.cat([hidden, hidden_pre, hidden_pos], dim=-1)
        x = self.fc2(x)
        return x

    def loss(self, input_ids, attention_mask, pre_label_all, pre_label, arg_label):

        with torch.no_grad():
            hidden = self.lm(input_ids, attention_mask)[0]

        hidden_pre = self._predict_predicate(hidden)
        loss_pre = self._att_ce_loss(hidden_pre, attention_mask, pre_label_all)

        pre_mask = convert_pre_bio_to_mask(pre_label)
        pre_pos_ids = self._get_position_ids(pre_mask, input_ids)
        hidden_pre_pos = self.POS(pre_pos_ids)
        hidden_pre = self._get_pre_hidden(pre_mask, hidden)
        hidden_arg = self._predict_argument(hidden, hidden_pre, hidden_pre_pos)
        loss_arg = self._att_ce_loss(hidden_arg, attention_mask, arg_label)

        return loss_pre, loss_arg

    @staticmethod
    def _get_position_ids(pre_mask, input_ids):
        position_ids = torch.zeros(pre_mask.shape, dtype=int, device=pre_mask.device)
        for mask_idx, cur_mask in enumerate(pre_mask):
            position_ids[mask_idx, :] += 2
            cur_nonzero = cur_mask.nonzero()
            start = torch.min(cur_nonzero).item()
            end = torch.max(cur_nonzero).item()
            position_ids[mask_idx, start:end + 1] = 1
            pad_start = max(input_ids[mask_idx].nonzero()).item() + 1
            position_ids[mask_idx, pad_start:] = 0
        return position_ids

    @staticmethod
    def _get_pre_hidden(pre_mask, hidden):
        B, L, D = hidden.shape
        hidden_pre = torch.zeros((B, L, D), device=pre_mask.device)
        for mask_idx, cur_mask in enumerate(pre_mask):
            pred_position = cur_mask.nonzero().flatten()
            pred_feature = torch.mean(hidden[mask_idx, pred_position], dim=0)
            pred_feature = torch.cat(L * [pred_feature.unsqueeze(0)])
            hidden_pre[mask_idx, :, :] = pred_feature
        return hidden_pre

    @staticmethod
    def _att_ce_loss(hidden, attention_mask, label):
        loss_fn = torch.nn.CrossEntropyLoss()
        n_pre_tag = hidden.size(-1)
        active_logits = hidden.view(-1, n_pre_tag)
        active_labels = torch.where(
            attention_mask.view(-1) == 1, label.view(-1),
            torch.tensor(loss_fn.ignore_index).type_as(label))
        loss = loss_fn(active_logits, active_labels)
        return loss


if __name__ == '__main__':
    config = BertSeqTagConfig(pos_embedding_dim=64,
                              fc_1_hidden_size=768, fc_1_dropout_rate=0.3, fc_2_hidden_size=768,
                              fc_2_dropout_rate=0.3, pretrained_model_name_or_path='bert-base-multilingual-cased')
    model = BertSeqTagModel(config).eval()
    model.load_state_dict(torch.load('../zh_model_159.pth', map_location=torch.device('cpu')), strict=False)
    text = ["微博内容不仅有哈士奇的照片，而且还配有生动的解说文字。",
            "大夏大学（英語：The Great China University）是上海的一所已被撤销的私立大学，1951年与光华大学合并成立华东师范大学。其旧址在今华东师范大学中山北路校区。大夏大学校友会于1985年1月20日在华东师范大学成立。华东师范大学将大夏大学建校日（6月1日）作为学校每年的纪念日。2017年，华东师范大学组建大夏书院。"]
    out = model(text)
    print(out)
