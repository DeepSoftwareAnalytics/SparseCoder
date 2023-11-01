# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import sys
import nltk
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
from rouge_metric import PyRouge
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import bleu
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          LongT5Config, LongT5ForConditionalGeneration, AutoTokenizer)
nltk.data.path.append('/3fs-jd/prod/deepseek/shared/guodaya/SparseCoder')
_ROUGE_EVAL = PyRouge(
    rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True, rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4
)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path):
        self.examples = pickle.load(open(file_path,"rb"))
        self.args = args
        self.tokenizer = tokenizer                     
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        source = self.examples[i]['file_contents']
        target = " ".join(self.examples[i]['file_docstring_tokens'])

        source_tokens = self.tokenizer.tokenize(source)[:self.args.max_source_length-2]
        source_tokens = [self.tokenizer.cls_token]+source_tokens+[self.tokenizer.sep_token]
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        source_ids += [self.tokenizer.pad_token_id]* (self.args.max_source_length - len(source_ids))

        target_tokens = self.tokenizer.tokenize(target)[:self.args.max_target_length-2]
        target_tokens = [self.tokenizer.cls_token]+target_tokens+[self.tokenizer.sep_token]
        target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
        target_ids += [self.tokenizer.pad_token_id]* (self.args.max_target_length - len(target_ids))

        return (
            torch.tensor(source_ids),
            torch.tensor(target_ids)
        )

def train(args, model, tokenizer):
    """ Train the model """
    #get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_filename)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)
    

    # Train!
    logger.warning("***** Running training *****")
    logger.warning("  Num examples = %d", len(train_dataset))
    logger.warning("  Num Epochs = %d", args.num_train_epochs)
    logger.warning("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.warning("  Total train batch size  = %d", args.train_batch_size)
    logger.warning("  Optimization steps per epoch = %d", len(train_dataloader))
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    model.train()
    losses, best_score = [], 0.0 
    for idx in range(args.num_train_epochs): 
        losses = []
        for step,batch in enumerate(train_dataloader):
            #get inputs
            source_ids, target_ids = [ x.to(args.device)  for x in batch]


            # forward and return loss
            loss = model(source_ids, target_ids)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            #report loss
            losses.append(loss.item())
            if len(losses)%100 == 0:
                logger.warning("epoch {} step {} loss {}".format(idx,step+1,round(np.mean(losses[-100:]),5)))
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 

            
        #evaluate    
        results = evaluate(args, model, tokenizer,args.dev_filename,  output_file_prefix = "dev",test_number = 1000)
        for key, value in results.items():
            logger.warning("  %s = %s", key, round(value,4))    
            
        #save best model
        if results['eval_score']>best_score:
            best_score = results['eval_score']
            logger.warning("  "+"*"*20)  
            logger.warning("  Best score:%s",round(best_score,4))
            logger.warning("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-score'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.warning("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer,file_name, test_number = 1000000, output_file_prefix = "test"):
    test_dataset = TextDataset(tokenizer, args, file_name)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size,num_workers=4)

    
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    losses = [] 
    gts = []
    preds = []

    for batch in test_dataloader:  
        source_ids, target_ids = [ x.to(args.device)  for x in batch]
        with torch.no_grad():
            loss = model(source_ids, target_ids)
            if args.n_gpu > 1:
                loss = loss.mean() 
            losses.append(loss.item())

            outs = model(source_ids)  
            for pred in outs:
                t=pred[0].cpu().numpy()
                t=list(t)
                if 0 in t:
                    t=t[:t.index(0)]
                text = tokenizer.decode(t,clean_up_tokenization_spaces=False).replace("</s>","")
                preds.append(text)
                if len(preds)%100 == 0:
                    logger.warning("evaluate {} samples".format(len(preds)))
        if len(preds) >= test_number:
            break
    preds = preds[:test_number]
    predictions  = []
    references = []
    gts = [" ".join(x['file_docstring_tokens']) for x in test_dataset.examples[:test_number]]
    with open(os.path.join(args.output_dir,"{}.output".format(output_file_prefix)),'w') as f, open(os.path.join(args.output_dir,"{}.gold".format(output_file_prefix)),'w') as f1:
        for idx,(pred,gold) in enumerate(zip(preds,gts)):
            predictions.append(str(idx)+"\t"+pred)
            references.append(str(idx)+"\t"+gold)
            f.write(pred+'\n')
            f1.write(gold+'\n')
        f.close()
        f1.close()     
    
    (goldMap, predictionMap) = bleu.computeMaps(predictions, references) 
    test_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)

    test_rouge = round(_ROUGE_EVAL.evaluate(preds, [[x] for x in gts])["rouge-l"]["f"] * 100,2)
    test_meteor = round(np.mean([nltk.translate.meteor_score.meteor_score([x.split()],y.split()) for x,y in zip(gts,preds)])*100,2)
    model.train()
    result = {
        "eval_loss":round(np.mean(losses),4),
        "eval_bleu":test_bleu,
        "eval_rouge":test_rouge,
        "eval_meteor":test_meteor,
        "eval_score": (test_bleu+test_rouge+test_meteor)/3
    }

    return result

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  

    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")  
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
    logger.warning(args)

    # Setup CUDA, GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count() 
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)
    args.device = device
    
   
    # Set seed
    set_seed(args.seed)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        
    
    config =LongT5Config.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.cls_token_id = 2
    tokenizer.sep_token_id = tokenizer.eos_token_id
    eos_ids = [tokenizer.eos_token_id]
    encoder = LongT5ForConditionalGeneration.from_pretrained(args.model_name_or_path,config=config)  

    model=Seq2Seq(encoder=encoder,decoder=encoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=eos_ids)
    

    if args.load_model_path is not None:
        logger.warning("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
    model.to(device)

    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        train(args, model, tokenizer)
       
    if args.do_test:
        results = evaluate(args, model, tokenizer,args.test_filename,  output_file_prefix = "test")
        for key, value in results.items():
            logger.warning("  %s = %s", key, round(value,4))    


                            

                
                
if __name__ == "__main__":
    main()


