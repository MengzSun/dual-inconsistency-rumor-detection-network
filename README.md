# dual-inconsistency-rumor-detection-network
Inconsistency Matters: A Knowledge-guided Dual-inconsistency Network for Multi-modal Rumor Detection  
Sun Mengzhu, Zhang Xi, Ma Jianqiang, Liu Yazheng  
https://aclanthology.org/2021.findings-emnlp.122   
EMNLP2021-findings

## Requirements
- Python 3.6
- torch 1.4.0
- transformers 4.9.1
- nltk


## Citation
If you use this code for your research, please cite our [paper](https://aclanthology.org/2021.findings-emnlp.122):
```
@inproceedings{sun-etal-2021-inconsistency-matters,
    title = "Inconsistency Matters: A Knowledge-guided Dual-inconsistency Network for Multi-modal Rumor Detection",
    author = "Sun, Mengzhu  and
      Zhang, Xi  and
      Ma, Jianqiang  and
      Liu, Yazheng",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.122",
    pages = "1412--1423",
    abstract = "Rumor spreaders are increasingly utilizing multimedia content to attract the attention and trust of news consumers. Though a set of rumor detection models have exploited the multi-modal data, they seldom consider the inconsistent relationships among images and texts. Moreover, they also fail to find a powerful way to spot the inconsistency information among the post contents and background knowledge. Motivated by the intuition that rumors are more likely to have inconsistency information in semantics, a novel Knowledge-guided Dual-inconsistency network is proposed to detect rumors with multimedia contents. It can capture the inconsistent semantics at the cross-modal level and the content-knowledge level in one unified framework. Extensive experiments on two public real-world datasets demonstrate that our proposal can outperform the state-of-the-art baselines.",
}

