# Encoder-Decoder Seq2Seq with Attention: English-Japanese Machine Translation
## エンコーダ・デコーダLSTMによるSeq2Seqによるアテンションを用いた日英翻訳
![](https://cdn-images-1.medium.com/max/2560/1*1I2tTjCkMHlQ-r73eRn4ZQ.png)

## Introduction

Statistical Machine Translation (SMT), which is the mainstream of machine translation, refers to a system that learns and translates into a target language a probabilistic model that maximizes the likelihood of parallel translation given the source language.    

Neural machine translation (NMT, Neural Machine Translation), which has become a topic of study for upgrading Google translation, is a method that uses neural networks as a probabilistic model to learn by statistical machine translation. Among them, the encoder-decoder translation model (Encoder-Decoder Model) can be translated only by neural networks and attracts a lot of attention.      

Neural networks can also use attention to act in the same way as we do on a part of the information given. For example, suppose you have an RNN that can consider the output of another RNN. This RNN is able to focus and focus on different positions of the output of the other RNN at each time step.      

## Technical Preferences

| Title | Detail |
|:-----------:|:------------------------------------------------|
| Environment | MacOS Mojave 10.14.3 |
| Language | Python |
| Library | Kras, scikit-learn, Numpy, matplotlib, Pandas, Seaborn |
| Dataset | [Tab-delimited Bilingual Sentence Pairs](http://www.manythings.org/anki/) |
| Algorithm | Encoder-Decoder LSTM |

## Refference

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [論文解説 Attention Is All You Need (Transformer)](http://deeplearning.hatenablog.com/entry/transformer)
- [How To Create Natural Language Semantic Search For Arbitrary Objects With Deep Learning](https://towardsdatascience.com/semantic-code-search-3cd6d244a39c)
- [TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- [Attentionで拡張されたRecurrent Neural Networks](https://deepage.net/deep_learning/2017/03/03/attention-augmented-recurrent-neural-networks.html)
