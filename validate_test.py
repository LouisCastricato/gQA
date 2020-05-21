import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from model_code.data_opinosis import *
from model_code.data_hotpotqa import local_num2text as gqa_local_num2text
from model_code.gSUM import gSUM
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
from pythonrouge.pythonrouge import Pythonrouge
from utils.beam_decoder import *
from utils.adamW import AdamW
from collections import Counter

def fetch_generated_summary(summary, oov, gqa=False):
    eos = 0
    try:
        eos = summaries.index(EOS)
    except:
        eos = len(summary)
    if gqa:
        return gqa_local_num2text(summary[0:eos], None, oov, test=True)
    return local_num2text(summary[0:eos], None, oov, test=True)


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

def summary_map(summaries, batch, gqa=False):
    return list(map(lambda i: " ".join(fetch_generated_summary(summaries[i], batch.oov[i], gqa)), range(len(summaries))))

def gold_map(batch):
    return list(map(lambda gold: " ".join(gold), batch.gold))

def gold_map_raw(strings):
    return list(map(lambda gold: " ".join(gold), strings))

def validate(dataset, run_model, model=None, \
    loss_function=None, pool=None, args=None, train_gqa=False):
    model.eval()

    #For when we run out of VRAM
    batch_size_old = dataset.args.batch
    dataset.args.batch = 5

    num_batches = (args.file_limit * 0.05)/dataset.args.batch 
    cur_cache = dataset.get_eval_item(0, args.cache, args.PG)
    itt_loss = 0

    for index in tqdm(range(1, int(num_batches), args.cache)):
        #print(index)
        async_result = pool.apply_async(dataset.get_eval_item, (index, args.cache, args.PG))
        #print(cur_cache)
        for batch in cur_cache:
            #Some batches have low data quality
            max_sample_size = int(max(batch.sent_sim.sum(dim=1)).item())
            model.change_sample_size(max_sample_size)
            loss, _, _ = run_model(model, batch, False, train_gqa)

            #Accumulate loss
            itt_loss += loss[0].data.sum()

        #Try to fetch the next cache, if it fails we stored a backup
        backup = cur_cache
        try:
            cur_cache = async_result.get()
        except:
            #If there was an issue with this batch, just load the next batch and continue
            cur_cache = backup
            continue

    model.change_sample_size(args.sample_size)
    dataset.args.batch = batch_size_old

    return itt_loss

def test_rouge(dataset, run_model, model=None, 
loss_function=None, pool=None, args=None, distributed = False):
    model.eval()

    #For when we run out of VRAM
    batch_size_old = dataset.args.batch
    dataset.args.batch = 1 
    is_coverage = (args.lambda_coeff > 0)

    num_batches = (args.file_limit * 0.05)/dataset.args.batch
    cur_cache = dataset.get_test_item(0, args.cache, args.PG)

    summary_text = []
    golden_text = []
    b = BeamSearch()
    if not distributed:
        for index in tqdm(range(1, int(num_batches), args.cache)):
            #print(index)
            async_result = pool.apply_async(dataset.get_test_item, (index, args.cache, args.PG))
            #print(cur_cache)
            for batch in cur_cache:
                #Some batches have low data quality
                returned_beam = b.beam_search([batch], model, args, is_coverage, run_model, False, beam_size=4)

                summary_text.append(summary_map([returned_beam.tokens[1:len(returned_beam.tokens)]], batch))
                golden_text.append(gold_map(batch))

            #Try to fetch the next cache, if it fails we stored a backup
            backup = cur_cache
            try:
                cur_cache = async_result.get()
            except:
                #If there was an issue with this batch, just load the next batch and continue
                cur_cache = backup
                continue
    else:
        for index in range(1, int(num_batches), args.cache):
            #print(index)
            async_result = pool.apply_async(dataset.get_test_item, (index, cpu_embedding, args.cache, args.PG))
            #print(cur_cache)
            for batch in cur_cache:
                #Some batches have low data quality
                returned_beam = b.beam_search([batch], model, args, False, run_model, False, beam_size=4)

                summary_text.append(summary_map([returned_beam.tokens[1:len(returned_beam.tokens)]], batch))
                golden_text.append(gold_map(batch))

            #Try to fetch the next cache, if it fails we stored a backup
            backup = cur_cache
            try:
                cur_cache = async_result.get()
            except:
                #If there was an issue with this batch, just load the next batch and continue
                cur_cache = backup
                continue        

    model.change_sample_size(args.sample_size)
    #print(golden_text)

    rouge = Pythonrouge(summary_file_exist=False,
                        summary=summary_text, reference=golden_text,
                        n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
                        recall_only=True, stemming=True, stopwords=True,
                        word_level=True, length_limit=False, length=args.gold_len,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)

    score = rouge.calc_score()
    if not distributed:
        print(score)

    model.change_sample_size(args.sample_size)
    dataset.args.batch = batch_size_old

    return score

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
    if args.cheat:
        cur_cache = dataset.get_training_item(0, args.cache)
    else:    
        cur_cache = dataset.get_test_item(0, args.cache)

    #Changing default params for testing
    pred_ans = list()
    pred_top_k = list()
    gold_ans = list()
    gold_top_k = list()
    b = BeamSearch()
    for index in tqdm(range(1, int(num_batches), args.cache)):
        if args.cheat:
            async_result = pool.apply_async(dataset.get_training_item, (index, args.cache))
        else:
            async_result = pool.apply_async(dataset.get_test_item, (index, args.cache))
        for batch in cur_cache:


            if args.span:
                output = run_model(model, batch, False, 
                args.PG, args.train_qa, tfr_override=0.0, span=args.span)


                for i in range(dataset.args.batch):
                    n_support = batch.hpqa_batch[i].ss_count
                    ss = output[1][i]
                    pred_ss = ((ss > 0.3).nonzero().cpu().detach().numpy()).flatten()
                    '''
                    print(pred_ss)
                    print(n_support)
                    print(torch.topk(output[1][i], k=n_support, dim=-1)[1].cpu().detach().numpy())
                    print(torch.topk(batch.hpqa_batch[i].sent_sim, k=n_support, dim=-1)[1].cpu().detach().numpy())
                    '''
                    #pred_ss = torch.topk(output[1][i], k=n_support, dim=-1)[1].cpu().detach().numpy()
                    gold_ss = torch.topk(batch.hpqa_batch[i].sent_sim, k=n_support, dim=-1)[1].cpu().detach().numpy()

                    text = find_span(batch.data_word_ex[i], 
                    output[2][i], output[3][i], output[4][i], 
                    batch.hpqa_batch[i].sent_count_no_q * batch.max_sent_len,
                    batch.hpqa_batch[i], dataset.args, mask=batch.data_mask_no_q[i])

                    pred_ans.append(text)
                    pred_top_k.append(pred_ss)

                    gold_ans.append(batch.gold[i])
                    gold_top_k.append(gold_ss)
                #sys.exit()
                continue
            num_supporting_sents = int(batch.sent_sim.sum(1).item())
            if args.hard_ss:
                num_supporting_sents = args.sample_size

            model.change_sample_size(num_supporting_sents)
            returned_beam = b.beam_search([batch], model, args, False, run_model, False, beam_size=8)
            pred_ans.append(summary_map([returned_beam.tokens[1:len(returned_beam.tokens)]], batch, True))
            gold_ans.append(gold_map(batch))
            batch.sent_sim[0].sum()
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
    if args.cheat:
        cur_cache = dataset.get_training_item(0, args.cache)
    else:    
        cur_cache = dataset.get_eval_item(0, args.cache)

    #Changing default params for testing
    pred_ans = list()
    pred_top_k = list()
    gold_ans = list()
    gold_top_k = list()
    
    b = BeamSearch()
    loss_total = 0

    for index in tqdm(range(1, int(num_batches), args.cache)):
        if args.cheat:
            async_result = pool.apply_async(dataset.get_training_item, (index, args.cache))
        else:
            async_result = pool.apply_async(dataset.get_eval_item, (index, args.cache))
        for batch in cur_cache:


            if args.span:
                output = run_model(model, batch, False, 
                args.PG, args.train_qa, tfr_override=0.0, span=args.span)
                
                loss = output[0].sum().cpu().data

                loss_total += loss
                for i in range(dataset.args.batch):
                    n_support = batch.hpqa_batch[i].ss_count
                    pred_ss = torch.topk(output[1][i], k=n_support, dim=-1)[1].cpu().detach().numpy()
                    gold_ss = torch.topk(batch.hpqa_batch[i].sent_sim, k=n_support, dim=-1)[1].cpu().detach().numpy()

                    text = find_span(batch.data_word_ex[i], 
                    output[2][i], output[3][i], output[4][i], 
                    batch.hpqa_batch[i].sent_count_no_q * batch.max_sent_len,
                    batch.hpqa_batch[i], dataset.args, mask=batch.data_mask_no_q[i])

                    pred_ans.append(text)
                    pred_top_k.append(pred_ss)

                    gold_ans.append(batch.gold[i])
                    gold_top_k.append(gold_ss)

                continue
            num_supporting_sents = int(batch.sent_sim.sum(1).item())
            #if args.hard_ss:
            #    max_sample_size = args.sample_size
            model.change_sample_size(num_supporting_sents)
            returned_beam = b.beam_search([batch], model, args, False, run_model, False, beam_size=4)
            pred_ans.append(summary_map([returned_beam.tokens[1:len(returned_beam.tokens)]], batch, True))
            gold_ans.append(gold_map(batch))
            batch.sent_sim[0].sum()
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

    model.change_sample_size(args.sample_size)
    dataset.args.batch = batch_size_old


    return accuracy, loss_total, N

def test(dataset, run_model, model, args=None):
    
    #For when we run out of VRAM
    batch_size_old = dataset.args.batch
    dataset.args.batch = 1 
    is_coverage = (args.lambda_coeff > 0)

    model.eval()
    b = BeamSearch()

    if args.cheat:
        batch = dataset.get_training_item(index=args.test_indx, delta=1)
    else:
        batch = dataset.get_eval_item(index=args.test_indx, delta=1)
    if args.train_qa:
        num_supporting_sents = int(batch[0].sent_sim.sum(1).item())
    
    #if args.hard_ss:
    #    max_sample_size = args.sample_size
    if args.train_qa:
        model.change_sample_size(num_supporting_sents)

    returned_beam = b.beam_search(batch, model, args, is_coverage, run_model, False, beam_size=4)
    #_, summary, _ = run_model(None, model, batch[0], True, 0, args.PG, train_gqa)

    print("Original Summary")
    print(batch[0].gold[0])
    print("Generated Summary")
    indx = 0

    #Clip at the first EOS
    try:
        indx = returned_beam.tokens.index(EOS)
    except:
        indx = len(returned_beam.tokens)
    try:
        if args.train_qa:
            print(gqa_local_num2text(returned_beam.tokens[1:indx], None, batch[0].oov[0], test=True))
        else:
            print(local_num2text(returned_beam.tokens[1:indx], None, batch[0].oov[0], test=True))
    except:
        print("Error! Word outside of OOV.")
        print(returned_beam.tokens[1:indx])
        print(batch[0].oov[0])

    if not args.train_qa:
        model.change_sample_size(args.sample_size)

    dataset.args.batch = batch_size_old
