# pytorch-OIE
Open Information Extraction

## Requirement

```
torch
transformers
tqdm
numpy
scikit-learn
```

## Quick Tour

监督的OIE任务被视为一种序列标注任务[[典型标注形式]](https://zhuanlan.zhihu.com/p/349699217)，从自然语句中抽取n-ary信息：
```
2009年11月，奥巴马 将对 中国 进行国事访问。
---------- ------      ---     -------
   ARG3     ARG0       ARG1     PRED

(奥巴马，国事访问，中国，2009年11月)
```

本仓库使用Bert-MLP-CLS作为基线模型进行信息抽取，更多模型选型优先参考引用。

### Inference
```python
from package.model.alpha import AlphaModel, AlphaConfig
import torch

config = AlphaConfig(pretrained_model_name_or_path='bert-base-multilingual-cased', pos_embedding_dim=64,
                     fc_1_hidden_size=768, fc_1_dropout_rate=0.3, fc_2_hidden_size=768, fc_2_dropout_rate=0.3,
                     pre_tag_size=3, arg_tag_size=9,)
model = AlphaModel(config).eval()
model.load_state_dict(torch.load('path_to_pth'))
text = ['enter your text']
out = model(text)
```

### Training
Like main_xxxx.py [[main_alpha.py]](main_alpha.py)

```python
import torch

from package.dataset.dense import Dataset, DataLoader
from package.model.beta import BetaModel, BetaConfig
from tqdm import tqdm

# definition
lr = 5e-3
num_workers = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = BetaConfig(pretrained_model_name_or_path='bert-base-multilingual-cased', pos_embedding_dim=64,
                    fc_1_hidden_size=768, fc_1_dropout_rate=0.2,
                    fc_2_hidden_size=768, fc_2_dropout_rate=0.2,
                    pre_tag_size=3, arg_tag_size=9
                    )
model = BetaModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

dataset = Dataset('./resource/OIE2016/train.oie.json', model.tokenizer)
dataloader = DataLoader(dataset, batch_size=128, collate_fn=dataset.collate_fn, shuffle=True, num_workers=num_workers)

# training
for i in range(200):
    with tqdm(total=len(dataloader), desc=f'Epoch {i}: Training ...') as t:
        for batch in dataloader:
            optimizer.zero_grad()
    
            input_ids, mask, pre_label_all, pre_label, arg_label = [_.to(device) for _ in batch]
            try:
                loss_pre, loss_arg = model.loss(input_ids, mask, pre_label_all, pre_label, arg_label)
            except:
                # skip noisy data
                continue
            loss = loss_pre + loss_arg
    
            loss.backward()
            optimizer.step()
    
            t.update()
            t.set_postfix(loss=loss.item(), loss_pre=loss_pre.item(), loss_arg=loss_arg.item())
    
    if (i + 1) % 25 == 0:
        torch.save(model.state_dict(), f"model_{i}.pth")
```
## Reference

- Supervised Open Information Extraction. NAACL-HLT. 2018. [[paper]](https://aclanthology.org/N18-1081/)
- Logician: A Unified End-to-End Neural Approach for Open-Domain Information Extraction. WSDM. 2018. [[paper]](https://doi.org/10.1145/3159652.3159712) [[zh-dataset-SAOKE]](https://ai.baidu.com/broad/introduction?dataset=saoke)
- Span Model for Open Information Extraction on Accurate Corpus. AAAI. 2020. [[paper]](https://aaai.org/ojs/index.php/AAAI/article/view/6497) [[github]](https://github.com/zhanjunlang/Span_OIE)
- Multi^2OIE: Multilingual Open Information Extraction Based on Multi-Head Attention with BERT. EMNLP. 2020. [[paper]](https://arxiv.org/abs/2009.08128) [[github]](https://github.com/youngbin-ro/Multi2OIE)

