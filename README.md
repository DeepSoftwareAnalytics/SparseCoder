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
lang=FCS
lr=2e-5
batch_size=16
beam_size=5
source_length=2048
target_length=128
output_dir=../saved_models/SparaseCoder
train_file=../../dataset/${lang}/train.pkl
dev_file=../../dataset/${lang}/dev.pkl
epochs=10 
pretrained_model=longformer-unixcoder #Roberta: roberta-base
mkdir -p $output_dir

#training and evaluating
python run.py \
--do_train \
--do_eval \
--model_name_or_path $pretrained_model \
--train_filename $train_file \
--dev_filename $dev_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--num_train_epochs $epochs 2>&1| tee $output_dir/train.log

#testing
reload_model=$output_dir/checkpoint-best-score/model.bin
test_file=../dataset/${lang}/test.pkl
python run.py \
--do_test \
--load_model_path $reload_model \
--model_name_or_path $pretrained_model \
--test_filename $test_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--num_train_epochs $epochs 2>&1| tee $output_dir/test.log
```

If you want to reproduce the other models mentioned in the paper, simply run the run.sh file under the appropriate folder. 

# Reference

If you use this code or SparseCoder, and if you use our FILE-CS dataset, please consider citing us.

```shell

```

