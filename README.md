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

## Reference

- Supervised Open Information Extraction. NAACL-HLT. 2018. [[paper]](https://aclanthology.org/N18-1081/)
- Logician: A Unified End-to-End Neural Approach for Open-Domain Information Extraction. WSDM. 2018. [[paper]](https://doi.org/10.1145/3159652.3159712) [[zh-dataset-SAOKE]](https://ai.baidu.com/broad/introduction?dataset=saoke)
- Span Model for Open Information Extraction on Accurate Corpus. AAAI. 2020. [[paper]](https://aaai.org/ojs/index.php/AAAI/article/view/6497) [[github]](https://github.com/zhanjunlang/Span_OIE)
- Multi^2OIE: Multilingual Open Information Extraction Based on Multi-Head Attention with BERT. EMNLP. 2020. [[paper]](https://arxiv.org/abs/2009.08128) [[github]](https://github.com/youngbin-ro/Multi2OIE)

