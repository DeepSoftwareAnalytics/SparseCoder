#CUDA_VISIBLE_DEVICES=0,1,3,4
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
