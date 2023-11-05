# SparseCoder

This repo will provide the code for reproducing the experiments in SparseCoder: Identifier-Aware Sparse Transformer for File-Level Code Summarization. 

SparseCoder employs a sliding window mechanism for self-attention to model short-term dependencies and leverages the structure of code to capture long-term dependencies among source code identifiers.


## Dependency
-pip install torch

-pip install transformers

## Data Download

Our file-level code summary dataset is released at hugging face. You can download the FILE-CS dataset at [this](https://huggingface.co/datasets/huangyx353/FILE-CS)

## Quick Tour 

git clone the repo first
```shell
git clone https://github.com/DeepSoftwareAnalytics/SparseCoder.git
```

After downloading the project, you can use SparseCoder to generate a file-level summary through the below request:
```shell
cd File-Leval-Summary/SparseCoder
bash run.sh
```

# Reference

If you use this code or SparseCoder, and if you use our FILE-CS dataset, please consider citing us.

```shell

```

