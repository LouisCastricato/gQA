import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from model_code.data_hotpotqa import local_num2text as gqa_local_num2text
from model_code.data_hotpotqa import *
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
from utils.adamW import AdamW
from collections import Counter


#Start = [indx, value]
#End = [indx, value]
class answer_pair:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __float__(self):
        return self.start[1].item() + self.end[1].item()
    def __str__(self):
        return str(self.start) + "\n" + str(self.end)


#Operates on a single batch. Returns the answer. Text length does NOT include the question.
def find_span(data_word_ex, start_logits, 
end_logits, a_type, text_length, data_point, args,
n_best = 128, mask=None):
    start_logits = start_logits.cpu().data.numpy()
    end_logits = end_logits.cpu().data.numpy()

    data_word_ex = data_word_ex.view(-1).cpu().data.numpy()
    a_type = int(torch.argmax(a_type).cpu().data)
    if a_type == 0:
        return 'yes'
    if a_type == 1:
        return 'no'
    start_best = np.argsort(-start_logits)[0:min(len(start_logits), n_best)]
    end_best = np.argsort(-end_logits)[0:min(len(end_logits), n_best)]
    pair = list()

    mask = mask.view(-1).cpu().data.numpy().tolist()

    for i in start_best:
        for j in end_best:
            start = [i, start_logits[i]]
            end = [j, end_logits[j]]

            if start[0] > end[0]:
                continue
            if start[0] > text_length:
                continue
            if end[0] > text_length:
                continue
            if end[0] - start[0] + 1 > args.gold_len:
                continue
            #Is there padding in our answer?
            if 0 in mask[start[0]:end[0]+1]:
                continue
            pair.append(answer_pair(start, end))

    pair.sort(key = lambda x: float(x), reverse=True)
    best = pair[0]

    text_toks = data_word_ex[best.start[0]:best.end[0]+1]
    return gqa_local_num2text(text_toks, None, data_point.oov, test=True)

def gold_map(batch):
    return list(map(lambda gold: " ".join(gold), batch.gold))

def gold_map_raw(strings):
    return list(map(lambda gold: " ".join(gold), strings))

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def update_sp(metrics, prediction, gold):
    cur_sp_pred = prediction
    gold_sp_pred = gold
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall


def test_qa(dataset, run_model, model=None, pool=None, args=None, samples=None):
    model.eval()

    #For when we run out of VRAM
    batch_size_old = dataset.args.batch
    if not args.span:
        dataset.args.batch = 1 

    args.cache = 10
    if samples == None:
        num_batches = (args.file_limit * 0.05)/dataset.args.batch
    else:
        num_batches = samples/dataset.args.batch
        args.cache=1

    cur_cache = dataset.get_test_item(0, args.cache)

    #Changing default params for testing
    pred_ans = list()
    pred_top_k = list()
    gold_ans = list()
    gold_top_k = list()

    for index in tqdm(range(1, int(num_batches), args.cache)):
        async_result = pool.apply_async(dataset.get_test_item, (index, args.cache))
        for batch in cur_cache:
            output = run_model(model, batch, False, args.train_qa, True, args.span)

            for i in range(dataset.args.batch):
                n_support = batch.hpqa_batch[i].ss_count
                ss = output[1][i]
                pred_ss = ((ss > 0.3).nonzero().cpu().detach().numpy()).flatten()
                gold_ss = torch.topk(batch.hpqa_batch[i].sent_sim, k=n_support, dim=-1)[1].cpu().detach().numpy()

                text = find_span(batch.data_word_ex[i], 
                output[2][i], output[3][i], output[4][i], 
                batch.hpqa_batch[i].sent_count_no_q * batch.max_sent_len,
                batch.hpqa_batch[i], dataset.args, mask=batch.data_mask_no_q[i])

                pred_ans.append(text)
                pred_top_k.append(pred_ss)

                gold_ans.append(batch.gold[i])
                gold_top_k.append(gold_ss)

        backup = cur_cache
        try:
            cur_cache = async_result.get()
        except:
            print("something hit me")
            cur_cache = backup
            continue
    num_exact_match = 0
    for i in range(len(gold_ans)):
        gold = gold_ans[i][0]
        pred = pred_ans[i][0]

        gold_ss = gold_top_k[i]
        pred_ss = pred_top_k[i]
        if gold.lower() == pred.lower():
            num_exact_match += 1
        
    accuracy = num_exact_match / len(gold_ans)
    print(f"final_accuracy: {accuracy} on {len(gold_ans)} samples")

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}

    for i in range(len(gold_ans)):
        em, prec, recall = update_answer(
            metrics, " ".join(pred_ans[i][0]), " ".join(gold_ans[i][0]))
    for i in range(len(gold_ans)):
        pred_ss = pred_top_k[i]
        gold_ss = gold_top_k[i]

        em, prec, recall = update_sp(
            metrics, pred_ss, gold_ss)



    N = len(gold_ans)
    for k in metrics.keys():
        metrics[k] /= N
    print("Metrics")
    print(metrics)
    dataset.args.batch = batch_size_old


    return accuracy


def validate_qa(dataset, run_model, model=None, pool=None, args=None, samples = 100):
    model.eval()
    random.shuffle(dataset.train)
    #For when we run out of VRAM
    batch_size_old = dataset.args.batch
    if not args.span:
        dataset.args.batch = 1 
    args.cache = 10

    num_batches = samples * args.cache
    cur_cache = dataset.get_eval_item(0, args.cache)

    #Changing default params for testing
    pred_ans = list()
    pred_top_k = list()
    gold_ans = list()
    gold_top_k = list()

    loss_total = 0

    for index in tqdm(range(1, int(num_batches), args.cache)):
        async_result = pool.apply_async(dataset.get_eval_item, (index, args.cache))
        for batch in cur_cache:
            output = run_model(model, batch, False, args.train_qa, True, args.span)
            
            loss = output[0].sum().cpu().data
            loss_total += loss

            for i in range(dataset.args.batch):
                n_support = batch.hpqa_batch[i].ss_count
                ss = output[1][i]
                pred_ss = ((ss > 0.3).nonzero().cpu().detach().numpy()).flatten()

                gold_ss = torch.topk(batch.hpqa_batch[i].sent_sim, k=n_support, dim=-1)[1].cpu().detach().numpy()

                text = find_span(batch.data_word_ex[i], 
                output[2][i], output[3][i], output[4][i], 
                batch.hpqa_batch[i].sent_count_no_q * batch.max_sent_len,
                batch.hpqa_batch[i], dataset.args, mask=batch.data_mask_no_q[i])

                pred_ans.append(text)
                pred_top_k.append(pred_ss)

                gold_ans.append(batch.gold[i])
                gold_top_k.append(gold_ss)

        backup = cur_cache
        try:
            cur_cache = async_result.get()
        except Exception as e: 
            print(e)
            print("Exception while validating! Oh no!")
            cur_cache = backup
            continue

    num_exact_match = 0
    for i in range(len(gold_ans)):
        gold = gold_ans[i][0]
        pred = pred_ans[i][0]
        if gold.lower() == pred.lower():
            num_exact_match += 1
    accuracy = num_exact_match / len(gold_ans)

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}

    for i in range(len(gold_ans)):
        em, prec, recall = update_answer(
            metrics, " ".join(pred_ans[i][0]), " ".join(gold_ans[i][0]))

    for i in range(len(gold_ans)):
        pred_ss = pred_top_k[i]
        gold_ss = gold_top_k[i]

        em, prec, recall = update_sp(
            metrics, pred_ss, gold_ss)    
    N = len(gold_ans)
    for k in metrics.keys():
        metrics[k] /= N
    print("Metrics")
    print(metrics)
    print("Average per example loss")
    print(loss_total/float(N))

    dataset.args.batch = batch_size_old


    return accuracy, loss_total, N