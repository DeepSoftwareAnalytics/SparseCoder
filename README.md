# SparseCoder

This repo will provide the code for reproducing the experiments in SparseCoder: Identifier-Aware Sparse Transformer for File-Level Code Summarization. 

SparseCoder employs a sliding window mechanism for self-attention to model short-term dependencies and leverages the structure of code to capture long-term dependencies among source code identifiers.


## Dependency
-pip install torch

-pip install transformers

## Data Download

Our file-level code summary dataset is released at hugging face. You can download the FILE-CS dataset at [this](https://huggingface.co/datasets/huangyx353/FILE-CS)

## Fine-Tuning
Here we provide fine-tuning settings of SparseCoder for file-level code summarization, whose results are reported in the paper.

```shell

```


# Reference

If you use this code or SparseCoder, and if you use our FILE-CS dataset, please consider citing us.

```shell

```

