import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.optim import RMSprop
import numpy as np
import random
import json
import time
import math
import sys
import os
from tqdm.autonotebook import tqdm
from multiprocessing.pool import ThreadPool
from torch.nn.utils import clip_grad_norm_
from warmup_scheduler import GradualWarmupScheduler


import sys
sys.path.append("..")
from utils.config import *
from utils.tfr_scheduler import tfr_scheduler
from utils.adamW import AdamW
from validate_test import *
from model_code.data_hotpotqa import *
from model_code.gQA_Span import gQA_Span

def to_var(tensors, cuda=True):
    if cuda:
        return [Variable(t, requires_grad=False).to(cuda_device)for t in tensors]
    else:
        return [Variable(t, requires_grad=False) for t in tensors]

def run_model(model, data, is_test = False, train_qa = False,
        decode = True, 
        span = False):


    
    if train_qa:
        question_word, question_ex, question_char, question_word_mask, question_char_mask, length_q = to_var(
        [data.question_word, data.question_word_ex, 
        data.question_char, data.question_word_mask, data.question_char_mask, data.length_q], cuda=args.cuda)

    data_word, data_ex, data_char, data_mask, length_s, extra_zeros = to_var(
        [data.data_word, data.data_word_ex, data.data_char, data.data_mask,
         data.length_s, data.extra_zeros], cuda=args.cuda)

    adj, gold_word, gold_mask_unk, gold_mask, gold_ex, all_to_all = to_var(
        [data.adjm, data.gold_word, data.gold_mask_unk, data.gold_mask, 
         data.gold_word_extended, data.all_to_all], cuda=args.cuda)
    try:
        sent_sim, sent_attn_mask = to_var(
            [data.sent_sim, data.sent_attn_mask], cuda=args.cuda)
    except:
        pass

    if args.span:
        if args.train_qa:
            answer_masks, answer_types, answer_start, answer_end, data_mask_no_q = to_var(
                [data.answer_masks, data.answer_types, data.answer_start, data.answer_end, data.data_mask_no_q])
        else:
            ext_class = to_var([data.ext_class])

    if train_qa:
        return model(data_word, data_char,
        data_mask, length_s, adj, all_to_all, sent_sim, sent_attn_mask,
        question_word, question_word_mask, question_char, question_char_mask, length_q,
        dev=cuda_device, start_pos=answer_start, end_pos=answer_end,
        a_type=answer_types, answer_mask=answer_masks, data_mask_no_q=data_mask_no_q)
            
    else:
        if span:
            return model(data_word, data_char, data_mask, 
            length_s, adj, all_to_all, sent_sim, sent_attn_mask, ext_class, cuda_device)

def save(model, optimizer, epoch, loss = None, scheduler=None, warmup=None):
    if scheduler is not None:
        obj = {'args':args, 'model':model.state_dict(), 'optimizer':optimizer.state_dict(), \
            'start': epoch+1, 'index' : 1,\
                'scheduler' : scheduler.state_dict(), 'warmup' : warmup.state_dict()}
    else:
        obj = {'args':args, 'model':model.state_dict(), 'optimizer':optimizer.state_dict(), \
            'start': epoch+1, 'index' : 1}        
    if loss is not None:
        loss_saved = np.loadtxt(open(args.save_path + "_loss_graph.csv", "rb"))
        if loss_saved is not None:
            loss_saved = np.append(np.asarray(loss_saved), loss)
        else:
            loss_saved = np.array(list(loss))
        #max because we use validation accuracy instead of loss
        best_loss = min(loss_saved)
        
        np.savetxt(args.save_path + "_loss_graph.csv", np.asarray(loss_saved))
        if loss == best_loss:
            print("Saving new best model.")
            torch.save(obj, args.save_path+'_best.model')
            if args.on_colab:
                print("Saving to google drive")
                torch.save(obj, '/content/drive/My Drive/Colab Notebooks/Gits/Graphical-Summarization/' + args.save_path + '_best.model')
    torch.save(obj, args.save_path+'.model')


