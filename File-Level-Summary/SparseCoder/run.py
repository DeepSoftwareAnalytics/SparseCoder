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
                          LongformerConfig, LongformerModel, RobertaTokenizer)
from lsg_converter import LSGConverter
converter = LSGConverter(max_sequence_length=2048)
nltk.data.path.append('/3fs-jd/prod/deepseek/shared/guodaya/SparseCoder')
_ROUGE_EVAL = PyRouge(
    rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True, rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4
)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARNING)
logger = logging.getLogger(__name__)
from tree_sitter import Language, Parser
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
parsers={}
for lang in ['python','ruby','java','go','javascript','php','c','cpp','c_sharp']:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    if lang == "c_sharp":
        parsers["c#"]= parser
    else:
        parsers[lang]= parser

_ROUGE_EVAL = PyRouge(
    rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True, rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4
)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

from tree_sitter import Language, Parser
assignment=['assignment','augmented_assignment']
function=['function_definition','class_definition']

import signal
def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")


def travel(node,tokenizer,tokens_index,parent_type=None):
    source_tokens = []
    global_mask = []
    identifier_mask = []
    if (len(node.children)==0 or node.type=='string' or node.type=='comment' or 'comment' in node.type):
        index = (node.start_point,node.end_point)
        code = tokens_index[index][1]
        tokens = tokenizer.tokenize('@ '+code)[1:]
        source_tokens += tokens
        if parent_type in ["import_statement","function_statement","assignment_statement"] and node.type=="identifier":
            if ["import_statement","function_statement"]:
                global_mask += [1]*min(len(tokens),1) +[0] * max((len(tokens)-1),0)
            else:
                global_mask += [2]*min(len(tokens),1) +[0] * max((len(tokens)-1),0)
        else:
            global_mask += [0] * len(tokens)
        if node.type != code:
            identifier_mask += [True]*min(len(tokens),1) +[False] * max((len(tokens)-1),0)
        else:
            identifier_mask = [False] * len(tokens)
        return source_tokens,global_mask,identifier_mask
    else:
        source_tokens = []
        global_mask = []
        identifier_mask =[]
        if node.type in ["import_statement"] or parent_type in ["import_statement"]:
            for child in node.children[:-1]:
                tmp = travel(child,tokenizer,tokens_index)   
                source_tokens += tmp[0]     
                global_mask += tmp[1] 
                identifier_mask += tmp[2]
            for child in node.children[-1:]:
                tmp = travel(child,tokenizer,tokens_index,"import_statement")   
                source_tokens += tmp[0]     
                global_mask += tmp[1]    
                identifier_mask += tmp[2]
        elif node.type in assignment or parent_type in ["assignment_statement"]:
            flag = True
            for child in node.children:
                if "=" in child.type:
                    flag = False
                tmp = travel(child,tokenizer,tokens_index,"assignment_statement" if flag else None)   
                source_tokens += tmp[0]     
                global_mask += tmp[1]  
                identifier_mask += tmp[2]  
        elif node.type in function or parent_type in ["function_statement"]:
            flag = False
            for child in node.children:
                tmp = travel(child,tokenizer,tokens_index,"function_statement" if flag else None)   
                source_tokens += tmp[0]     
                global_mask += tmp[1]  
                identifier_mask += tmp[2]         
                if child.type == "def" or child.type == "class":
                    flag = True
                else:
                    flag = False                    
        else:
            for child in node.children:
                tmp = travel(child,tokenizer,tokens_index)   
                source_tokens += tmp[0]     
                global_mask += tmp[1]
                identifier_mask += tmp[2]
        #if any([x in node.type for x in ["statement","definition","block",'assignment']]):
        #    identifier_mask[-1] = True
        return  source_tokens, global_mask,identifier_mask  



def parse(code,tokenizer,lang):
    parser = parsers[lang]
    root_node = parser.parse(bytes(code,'utf8')).root_node
    #print(root_node.children[0].children)
    #exit()
    tokens_index=tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x,code) for x in tokens_index]  
    index_to_code = {}
    for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
        index_to_code[index] = (idx,code)  
    source_tokens,global_mask,identifier_mask = travel(root_node,tokenizer,index_to_code)
    #print(source_tokens)
    #print(global_mask)
    #print(root_node.sexp())
    #exit()
    return source_tokens,global_mask,identifier_mask


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path):
        self.examples = []
        with open(file_path,"r") as f:
            for line in f:
                js = json.loads(line)
                js['file_contents'] = " ".join(js["code_tokens"])
                js['file_docstring_tokens'] = js['docstring_tokens']
                self.examples.append(js)
        self.args = args
        self.tokenizer = tokenizer                     
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        #source = self.examples[i]['file_contents']
        target = " ".join(self.examples[i]['file_docstring_tokens'])
        '''
        code = """
        import re as kak
        a,b = 0,1
        abc += a
        def func():
            pass
        class c1():
            pass
        """
        '''

        code = self.examples[i]['file_contents']
        source_tokens,global_mask,identifier_mask = parse(code,self.tokenizer,self.examples[i]["language"])
        assert len(source_tokens) == len(global_mask) and len(source_tokens) == len(identifier_mask)

        source_tokens = [self.tokenizer.cls_token,"<encoder-only>",self.tokenizer.sep_token] + source_tokens[:self.args.max_source_length-4] + [self.tokenizer.sep_token]
        identifier_mask = [False,False,False] + identifier_mask[:self.args.max_source_length-4] + [False]
        global_mask = [1,0,0] + global_mask[:self.args.max_source_length-4] + [0]

        cont = 0
        for p in range(1,3):
            for i in range(len(global_mask)):
                if cont == self.args.max_global_length or global_mask[i] == 0:
                    if global_mask[i] is not True:
                        global_mask[i] = False
                elif global_mask[i] is True:
                    continue
                elif global_mask[i] == p:
                    global_mask[i] = True
                    cont += 1

        cont = 0
        for i in range(len(identifier_mask)):
            if cont == self.args.max_identifier_length or global_mask[i]:
                identifier_mask[i] = False
            elif identifier_mask[i]:
                cont += 1

        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        source_ids += [self.tokenizer.pad_token_id]* (self.args.max_source_length - len(source_ids))   
        global_mask += [False]* (self.args.max_source_length - len(global_mask))      
        identifier_mask += [False]* (self.args.max_source_length - len(identifier_mask))  

        target_tokens = self.tokenizer.tokenize(target)[:self.args.max_target_length-2]
        target_tokens = [self.tokenizer.cls_token]+target_tokens+[self.tokenizer.sep_token]
        target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
        target_ids += [self.tokenizer.pad_token_id]* (self.args.max_target_length - len(target_ids))
         #print(len(source_ids),len(global_mask),len(target_ids))
        return (
            torch.tensor(source_ids),
            torch.tensor(global_mask),
            torch.tensor(identifier_mask),
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
            source_ids, global_mask,identifier_mask, target_ids = [ x.to(args.device)  for x in batch]


            # forward and return loss
            loss = model(source_ids,global_mask,identifier_mask,target_ids)

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
        results = evaluate(args, model, tokenizer,args.dev_filename, test_number = 1000, output_file_prefix = "dev")
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


def evaluate(args, model, tokenizer,file_name, test_number = 10000000, output_file_prefix = "test"):
    test_dataset = TextDataset(tokenizer, args, file_name)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size,num_workers=4)

    
    # Eval!
    logger.warning("***** Running evaluation *****")
    logger.warning("  Num examples = %d", len(test_dataset))
    logger.warning("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    losses = [] 
    gts = []
    preds = []

    for batch in test_dataloader:  
        source_ids,global_mask,identifier_mask, target_ids = [ x.to(args.device)  for x in batch]
        with torch.no_grad():
            loss = model(source_ids,global_mask,identifier_mask, target_ids)
            if args.n_gpu > 1:
                loss = loss.mean() 
            losses.append(loss.item())

            outs = model(source_ids,global_mask,identifier_mask)  
            for pred in outs:
                t=pred[0].cpu().numpy()
                t=list(t)
                if 0 in t:
                    t=t[:t.index(0)]
                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                preds.append(text.lower())
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
    parser.add_argument("--max_global_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_identifier_length", default=512, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
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
        


    config = LongformerConfig.from_pretrained(args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config.attention_window = [512 for x in config.attention_window]
    #budild model
    encoder = LongformerModel.from_pretrained(args.model_name_or_path,config=config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,tokenizer=tokenizer,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    total_params = 0 
    params = 0
    for n,p in model.named_parameters():
        if any([x in n for x in ["lora"]]):
            p.requires_grad = True
            if "out" in n:
                torch.nn.init.zeros_(p)
            total_params += p.numel()
            print(n)
        params += p.numel()
    logging.info('Total number of lora parameters: {}M, rate: {}%'.format(total_params//1000/1000,round(total_params/params*100,2)))

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


