import argparse
import string
import json
import re
import random
import spacy
import torch
import numpy as np
import scipy.sparse as sp
import pickle
import torchtext
import os
import sys

from functools import reduce
from torch import nn

#Used for attention forcing. Soft cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


#Convert data to states
import sys
sys.path.append("..")
from utils.state import State
from utils.config import *

nlp = spacy.blank('en')

'''
final experiment
change to spacy tokenization (preprocess more)
change adjm from cos sim to
every sent connect to next sent, last sent in doc connect to first sent in doc
'''

count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer = CountVectorizer()

D_WORD_EMB = args.d_word_embed

vocab = ' ' + string.ascii_letters + string.digits + string.punctuation

character2id = {c:idx for idx, c in enumerate(vocab)}
UNK_CHAR = len(character2id)
CHAR_VOCAB_SIZE = len(vocab) + 1

#Pulled directly from PGMMR. Vocab size there was 50000
WORD_VOCAB_SIZE = 50000
SOS = 0
EOS = 1
PAD_WORD = 2
UNK_WORD = 3

word2id = {'SOS' : 0, 'EOS' : 1, 'PAD': PAD_WORD, 'UNK': UNK_WORD}
id2word = ['SOS', 'EOS', 'PAD', 'UNK']

embedding = torch.zeros((WORD_VOCAB_SIZE, D_WORD_EMB))
vocab = torchtext.vocab.GloVe(name='6B', dim=D_WORD_EMB)
word_count = WORD_VOCAB_SIZE

flatten = lambda l: [item for sublist in l for item in sublist]


def load_word_vocab(vocab_file):
    sorted_word = json.load(open(vocab_file))
    word_count = min(WORD_VOCAB_SIZE, len(sorted_word))
    words = [t[0] for t in sorted_word[:word_count]]
    global word2id
    i = 4
    for w in words:
        word2id[w.lower()] = i
        i += 1
        #id2word.append(w.lower())
        if i == WORD_VOCAB_SIZE:
            break
    cnt = 0
    for w,_id in word2id.items():
        if w not in vocab.stoi:
            cnt += 1
        embedding[_id] = vocab[w]
    global id2word
    id2word = {v: k for k, v in word2id.items()}
    print("%d words don't have pretrained embedding" % cnt)


def get_word_id(w):
    w = w.lower()
    return word2id[w] if w in word2id else UNK_WORD

def get_char_id(c):
    return character2id[c] if c in character2id else UNK_CHAR

def repeat(f, n):
    def rfun(p):
        return reduce(lambda x, _: f(x), range(n), p)
    return rfun

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.maximum(np.array(adj.sum(1)), 1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def get_gold_embedding(arr):
    gold_embedding = np.zeros((len(arr[0]), D_WORD_EMB))
    for elem, _id in arr[0]:
        gold_embedding[_id] = vocab[elem]
    return gold_embedding

def flattenmax(inp_arr):
    to_process = [s for s in inp_arr]
    return(max(map(lambda x: len(flatten(x)), to_process)))

def vanilla_tokenize(sentence):
    tok_sentence = re.findall(r"[\w]+[']*[\w]+|[\w]+|[.,!?;]", sentence)
    return tok_sentence

def spacy_tokenize(sentence):
    doc = nlp(sentence)
    return [token.text for token in doc]

def local_num2text(inp, length, oov, test=False):
    to_return = []
    for indx in inp:
        int_indx = int(indx)
        if int_indx >= WORD_VOCAB_SIZE:
            to_return.append(oov[int_indx - WORD_VOCAB_SIZE])
        else:
            to_return.append(id2word[int_indx])
        if to_return[-1] == "EOS" and test:
            #Try just incase to_return is only one character. In that case we still want to see the EOS
            try:
                return to_return[0:len(to_return) - 1]
            except:
                continue
    if test:
        return to_return
    else:
        return to_return[0:int(length)]

def read_hpqa_files(path, files, args):
    count = 0
    batch_count = 0
    batch_id = 0

    batches = []
    cur_batch = []
    for hpqa in files:
        file = hpqa.name
        with open(os.path.join(path, file)) as json_file:
            data = json.load(json_file)
            cur_batch.append(HotPotQA(data, args, True))
        count += 1
        batch_count += 1
        if batch_count >= args.batch:
            batch = Batch(cur_batch, batch_id)
            batches.append(Batch(cur_batch, batch_id))
            batch_id += 1
            batch_count = 0
            cur_batch = []
    return batches


class hpqa_short_hand():
    def __init__(self, name, id):
        self.name = name
        self.id = id


'''
need attention forcing -> add code -> read overleaf -> positive attention forcing with supporting sentences, overleaf describes negative attention forcing
word_data incorrect (trailing values should be pad)
char_mask good
char_data good -> get rid of 
answer_word_extended  (should have same value, if it has oov, has oov from original data)
    sos answer eos pad pad ... 15
answer_mask incorrect (just look at it)
answer_mask: (1 for all non pad terms)
answer_mask_unk: (1 if word in oov)


the thing with gQA, attention forcing
if gSUM gets accepted, we have gQA, they are helping each other.
'''

'''
answer_word_extended  (should have same value, if it has oov, has oov from original data)
    sos answer eos pad pad ... 15

change answer to gold
'''
class HotPotQA():
    #Articles to be given as a disjoint set of sentences
    def compute_span(self, articles, gold):
        start = [0,0]
        end = [0,0]
        if gold[0] in ['yes', 'no']:
            return gold[0]
        else:
            index = 0
            for i, sentence in enumerate(articles):
                for j, word in enumerate(sentence):
                    if word == gold[index]:
                        if index == 0:
                            start = [i,j]
                        
                        #Are we at the end of the phrase
                        index += 1
                        if index >= len(gold):
                            end = [i,j]
                            return start, end
                    else:
                        index = 0

        return "Not Found"

    def __init__(self, obj, args=None, pointer_generation=False):
        self.args=args

        self.id = obj['id']

        self.name = str(obj['id']) + ".json"

        self.gold = obj['answer'] # now tokenized and lower()'ed at preproc step

        self.question = obj['question'] # now tokenized and lower()'ed at preproc step

        self.articles = obj['article_sentences']

        self.article_tokens = obj['article_sentence_tokens']

        self.sentences_flat = list()
        for article in self.articles:
            self.sentences_flat.extend(article)
        self.sent_count_no_q = len(self.sentences_flat)
        
        self.sentences_flat.append(' '.join(self.question))

        self.sentence_tokens_flat = list()
        for article_token in self.article_tokens:
            self.sentence_tokens_flat.extend(article_token)
        #Store a version with no question attached
        self.sentence_tokens_flat_no_q =self.sentence_tokens_flat


        #Find Span
        self.start, self.end = [[0,-100],[0,-100]]
        
        self.answer_type = 2
        self.answer_mask = int(self.answer_type / 2.0)

        
        span = self.compute_span(self.sentence_tokens_flat_no_q, self.gold)

        if span == "yes":
            self.answer_type = 0
        elif span == "no":
            self.answer_type = 1
        else:
            start, end = span
                
        #Unless answer type = 2, just mask

        if self.answer_type == 2:
            #Manage windowing. Keep expanding the window around the answer until we can no longer do so
            if end[1] >= args.max_sentence_len or start[1] >= args.max_sentence_len:
                temp = ""

                length = (end[1] + 1 - start[1])



                for i in range(args.max_sentence_len // 2):
                    if length > args.max_sentence_len:
                        #If the length of the answer is longer than the sentence. clip gold
                        temp = self.sentence_tokens_flat_no_q[start[0]][start[1]:args.max_sentence_len + start[1]]
                        self.gold = temp
                        break
                    temp = \
                        self.sentence_tokens_flat_no_q[start[0]][(start[1] - i):(end[1] + i + 1)]
                    #Finish the expansion in whatever direction didnt hamper us
                    delta = args.max_sentence_len - len(temp)
                    #Go back a step, we went too far
                    if delta < 0:
                        i -= 1
                        temp = \
                            self.sentence_tokens_flat_no_q[start[0]][(start[1] - i):(end[1] + i + 1)]
                        break
                    #We are at maximum length
                    if delta == 0:
                        break

                    #We hit the right boundary
                    if (end[1] + i + 1) >= len(self.sentence_tokens_flat_no_q[start[0]]):
                        temp = \
                            self.sentence_tokens_flat_no_q[start[0]][max(0,(start[1] - i - delta)):]
                        break
                    #We hit the left boundary
                    if (start[1] - i) <= 0:
                        temp = \
                            self.sentence_tokens_flat_no_q[start[0]][:(end[1] + i + 1 + delta)]
                        break                                    
                self.sentence_tokens_flat_no_q[start[0]] = temp

                start, end = self.compute_span(self.sentence_tokens_flat_no_q, self.gold[:min(len(self.gold), args.max_sentence_len)])

            self.start, self.end = start, end
        
        #If we changed windowing, update here
        self.sentence_tokens_flat = self.sentence_tokens_flat_no_q

        #The question is to be taken into account for the GCN
        self.sentence_tokens_flat.append(self.question)

        self.supporting_sentences = obj['flatten_supporing_sentences']
        self.ss_count = int(np.sum(self.supporting_sentences))
        #We can use the question in the answer itself
        if not args.span:
            self.supporting_sentences.append(1)

        # gqa mark3, create adjm on the fly

        # with open(adjm_dir + str(self.id) + ".adjm", 'r') as open_file:
        #     sub_adjm = json.loads(open_file.read())
        # sub_adjm = torch.FloatTensor(sub_adjm)

        # number of sentences total in this data point
        self.sent_count = len(self.sentences_flat)

        # max number of words in a sentence
        self.max_sent_len = 0
        for sent_tokens in self.sentence_tokens_flat:
            self.max_sent_len = max(self.max_sent_len, len(sent_tokens))
        self.max_sent_len = min(self.max_sent_len, args.max_sentence_len)

        self.max_sent_len_no_q = 0
        for sent_tokens in self.sentence_tokens_flat_no_q:
            self.max_sent_len_no_q = max(self.max_sent_len_no_q, len(sent_tokens))
        self.max_sent_len_no_q = min(self.max_sent_len_no_q, args.max_sentence_len)


        # max number of characters in a word
        self.max_word_len = 0
        for sent_tokens in self.sentence_tokens_flat:
            for token in sent_tokens:
                self.max_word_len = max(self.max_word_len, len(token))
        self.max_word_len = min(self.max_word_len, args.max_word_len)
        # question_length
        self.question_len = len(self.question)

        # Account for OOV words
        self.oov = list()

        #########################
        ### construct data points for question
        #########################
        question_char = np.full((self.question_len, self.max_word_len), 0, dtype=int)
        question_char_mask = np.zeros((self.question_len, self.max_word_len), dtype=int)
        question_word = np.full((self.question_len), PAD_WORD, dtype=int)
        question_word_mask = np.full((self.question_len), 0, dtype=int)
        question_word_extended = np.full((self.question_len), PAD_WORD, dtype=int)

        # test for num questions with > 64 words
        num_question_over_words = 0
        for i, q_token in enumerate(self.question):
            if i >= args.max_question_len:
                num_question_over_words += 1
                print(question_tokens)
                break
            question_word[i] = get_word_id(q_token)
            question_word_extended[i] = question_word[i]
            question_word_mask[i] = 1
            if question_word_extended[i] == UNK_WORD:
                if q_token not in self.oov:
                    self.oov.append(q_token)
                question_word_extended[i] = WORD_VOCAB_SIZE + self.oov.index(q_token)

            for j, q_char in enumerate(q_token):
                if j >= self.max_word_len:
                    break
                question_char[i][j] = get_char_id(q_char)
                question_char_mask[i, j] = 1
        if num_question_over_words:
            print(f"WARNING: number of questions with over {args.max_question_len} words: {num_question_over_words}")

        length_q = np.array([self.question_len])
        self.question_char = torch.from_numpy(question_char)
        self.question_char_mask = torch.from_numpy(question_char_mask)
        self.question_word = torch.from_numpy(question_word)
        self.question_word_mask = torch.from_numpy(question_word_mask)
        self.question_word_extended = torch.from_numpy(question_word_extended)

        self.length_q = torch.from_numpy(np.array(len(self.question)))

        #########################
        ### construct data points documents
        #########################

        ### construct mask for character level (num sentences x num words x num characters 3d array)
        data_char = np.full((self.sent_count, self.max_sent_len, self.max_word_len), 0, dtype=int)
        data_char_mask = np.zeros((self.sent_count, self.max_sent_len, self.max_word_len), dtype=int)

        ### construct mask for word level (num sentences x num words 2d array) one/zeros
        data_word = np.full((self.sent_count, self.max_sent_len), PAD_WORD, dtype=int)
        data_word_mask = np.full((self.sent_count, self.max_sent_len), 0, dtype=int)
        data_word_mask_no_q = np.full((self.sent_count - 1, self.max_sent_len), 0, dtype=int)
        data_word_extended = np.full((self.sent_count, self.max_sent_len), PAD_WORD, dtype=int)

        for i, sent_words in enumerate(self.sentence_tokens_flat):
            for j in range(len(sent_words)):
                if j >= self.max_sent_len:
                    break
                data_word_mask[i, j] = 1
                if i != len(self.sentence_tokens_flat) - 1:
                    data_word_mask_no_q[i, j] = 1
                data_word[i, j] = get_word_id(sent_words[j])
                
                # copy extended vocab for PG
                data_word_extended[i, j] = data_word[i, j]
                if data_word_extended[i, j] == UNK_WORD:
                    if sent_words[j] not in self.oov:
                        self.oov.append(sent_words[j])
                    data_word_extended[i, j] = WORD_VOCAB_SIZE + self.oov.index(sent_words[j])

                for k, c in enumerate(sent_words[j]):
                    if k >= self.max_word_len:
                        continue
                    else:
                        data_char_mask[i][j][k] = 1
                        data_char[i, j, k] = get_char_id(c)

        #########################
        ### construct adjacency matrix
        #########################

        article_adjms = list() # what do you think this is lol
        total_len = 0

        for article_idx, article in enumerate(self.articles):
            ind = np.diag_indices(len(article))
            article_adjm = torch.ones(len(article), len(article))
            article_adjm[ind[0], ind[1]] = 0 # zero out diag
            article_adjms.append(article_adjm)

            total_len += len(article)

        ind = np.diag_indices(total_len)

        doc_adjm = torch.zeros(total_len + 1, total_len + 1)
        start_idx = 0
        end_idx = 0
        for adjm_idx, adjm in enumerate(article_adjms):
            end_idx += len(adjm)
            doc_adjm[start_idx:end_idx, start_idx:end_idx] = adjm
            start_idx += len(adjm)

        rot_ind = np.diag_indices(total_len - 1)
        doc_adjm[rot_ind[0], rot_ind[1] + 1] = 1
        doc_adjm[-2][-1] = 1
        
        doc_adjm[-1] = 1 # question augment properly
        doc_adjm[:,-1] = 1


        self.adjm = doc_adjm 
        
        #########################
        ### construct data points for answer
        #########################


        '''
        print(self.gold)
        if span not in ["Not Span", "Not Found"]:
            start, end = span
            print(self.sentence_tokens_flat_no_q[start[0]][start[1]:end[1]])
        print(span)
        '''

        gold_word = np.zeros(len(self.gold) + 2, dtype = int) # literally answer word encodings
        gold_word_extended = np.zeros(len(self.gold)  + 2, dtype = int) # above, with oov's
        gold_word_mask = np.full(len(self.gold) + 2, 1, dtype = int) # mask for non-oov words
        gold_word_mask_unk = np.full(len(self.gold) + 2, 0, dtype = int) # mask for oov words

        gold_word[0] = SOS
        gold_word_extended[0] = SOS

        for i in range(len(self.gold)):
            gold_word[i + 1] = get_word_id(self.gold[i].lower())
            gold_word_extended[i + 1] = get_word_id(self.gold[i].lower())

        gold_word[-1] = EOS
        gold_word_extended[-1] = EOS

        #########################
        ### construct data points for similarity score
        #########################
        # sent_sim = np.zeros(self.max_sent_len, dtype = float)
        ##### experiment 2019/09/11
        # sentence_cat_questions = [s.lower() + " " + ' '.join(self.question) for s in self.sentences_flat]
        # sentence_cat_question_vectors = count_vectorizer.fit_transform(sentence_cat_questions)
        # sentence_vectors = count_vectorizer.transform(self.sentences_flat)
        # self.all_to_all = cosine_similarity(sentence_cat_question_vectors, sentence_vectors)
        ##### experiment is to use rotated diagonal, with end of doc -> begin of doc

        article_adjms = list() # what do you think this is lol
        total_len = 0

        for article_idx, article in enumerate(self.articles):
            ind = np.diag_indices(len(article))
            article_adjm = torch.zeros(len(article), len(article))
            article_adjm[-1, 0] = 1
            article_adjms.append(article_adjm)

            total_len += len(article)

        ind = np.diag_indices(total_len)

        doc_adjm = torch.zeros(total_len + 1, total_len + 1)
        start_idx = 0
        end_idx = 0
        for adjm_idx, adjm in enumerate(article_adjms):
            end_idx += len(adjm)
            doc_adjm[start_idx:end_idx, start_idx:end_idx] = adjm
            start_idx += len(adjm)

        rot_ind = np.diag_indices(total_len - 1)
        doc_adjm[rot_ind[0], rot_ind[1] + 1] = 1
        
        doc_adjm[-1] = 1 # question augment properly
        doc_adjm[:,-1] = 1
        self.all_to_all = doc_adjm

        self.total_len = total_len



        # print(f"article_vectors: {article_vectors}")
        # print(f"question_vectors: {question_vectors}")

        ### list of sentence lengths
        self.length_s = np.array(list(map(lambda sentence: min(args.max_sentence_len, sentence), data_word_mask.sum(axis=1))))
        self.max_sent_len = max(self.length_s)
        self.length_s = torch.from_numpy(self.length_s)
        
        word_arr = list(map(lambda i: local_num2text(data_word_extended[i], self.length_s[i], self.oov), 
        range(len(data_word_extended))))

        words = list(map(lambda strings: " ".join(strings), word_arr))

        documents_count = count_vectorizer.fit_transform(words)
        sent_sim = cosine_similarity(documents_count, documents_count)

        del documents_count


        self.adjm += 0.5 * (sent_sim - np.eye(sent_sim.shape[0]))

        #Make matrix symmetric
        self.adjm = (0.5) * (torch.t(self.adjm) + self.adjm)
        
        ### Attention forcing
        self.sent_sim = np.array(self.supporting_sentences).astype(int)
        self.sent_sim = torch.from_numpy(self.sent_sim)
        self.sent_attn_mask = torch.full(self.sent_sim.size(), 1)

        # sent_attn_mask_pos = list(map(lambda x: 1 if x > (1. - s_mask) else 0, sent_sim))

        self.gold_word = torch.from_numpy(gold_word)
        self.gold_word_extended = torch.from_numpy(gold_word_extended)
        self.gold_mask = torch.from_numpy(gold_word_mask)
        self.gold_mask_unk = torch.from_numpy(gold_word_mask_unk)

        self.data_char = torch.from_numpy(data_char)
        self.data_word = torch.from_numpy(data_word)
        self.data_word_mask = torch.from_numpy(data_word_mask)
        self.data_word_mask_no_q = torch.from_numpy(data_word_mask_no_q)
        self.data_word_extended = torch.from_numpy(data_word_extended)

        # self.all_to_all = np.full((self.sent_count, self.sent_count), 1., dtype=float) - np.eye(self.sent_count)
        # self.all_to_all = torch.from_numpy(self.all_to_all)

class Batch:
    def __init__(self, hpqa_batch, _id):
        self.id = _id
        self.hpqa_batch = hpqa_batch
        self.batch_size = len(hpqa_batch)
        self.max_num_sent = max([x.sent_count for x in self.hpqa_batch])
        self.max_question_len = max([x.question_len for x in self.hpqa_batch])
        self.max_sent_len = max([x.max_sent_len for x in self.hpqa_batch])
        self.max_word_len = max([x.max_word_len for x in self.hpqa_batch])
        self.max_gold_len = max([len(x.gold_word) for x in self.hpqa_batch])
        self.max_oov_len = max([len(x.oov) for x in self.hpqa_batch])
        self.max_length_q = max(x.question_len for x in self.hpqa_batch)
        
        self.oov = [x.oov for x in self.hpqa_batch]
        self.extra_zeros = None
        if self.max_oov_len > 0:
            self.extra_zeros = torch.zeros((self.batch_size, self.max_oov_len))

        if hpqa_batch[0].args.span:
            answer_masks = list()
            answer_types = list()
            answer_start = list()
            answer_end   = list()

            for x in self.hpqa_batch:
                answer_masks.append(x.answer_mask)
                answer_types.append(x.answer_type)
                
                #Stride
                answer_start.append(x.start[0] * self.max_sent_len + x.start[1])
                answer_end.append(x.end[0] * self.max_sent_len + x.end[1])
            
            self.answer_masks = torch.LongTensor(answer_masks)
            self.answer_types = torch.LongTensor(answer_types)
            self.answer_start = torch.LongTensor(answer_start)
            self.answer_end   = torch.LongTensor(answer_end)

        self.data_mask_no_q = torch.stack([
                torch.IntTensor(np.pad(x.data_word_mask_no_q, 
                    pad_width=[(0, self.max_num_sent - x.sent_count_no_q), (0, self.max_sent_len-x.max_sent_len_no_q)], 
                    mode='constant', constant_values=0))
                for x in self.hpqa_batch])

        self.question_word = torch.stack([
                torch.LongTensor(np.pad(x.question_word, 
                    pad_width=[(0, self.max_question_len-x.question_len)], mode='constant', constant_values=PAD_WORD))
                for x in self.hpqa_batch])

        self.question_word_ex = torch.stack([
                torch.LongTensor(np.pad(x.question_word_extended, 
                    pad_width=[(0, self.max_question_len-x.question_len)], mode='constant', constant_values=PAD_WORD))
                for x in self.hpqa_batch])

        self.question_word_mask = torch.stack([
                torch.LongTensor(np.pad(x.question_word_mask, 
                    pad_width=[(0, self.max_question_len-x.question_len)], mode='constant', constant_values=0))
                for x in self.hpqa_batch])

        self.question_char = torch.stack([
                torch.LongTensor(np.pad(x.question_char,
                    pad_width=[(0, self.max_question_len-x.question_len), (0, self.max_word_len-x.max_word_len)],
                    mode='constant', constant_values=0))
                for x in self.hpqa_batch])

        self.question_char_mask = torch.stack([
                torch.LongTensor(np.pad(x.question_char_mask, 
                    pad_width=[(0, self.max_question_len-x.question_len), (0, self.max_word_len-x.max_word_len)], 
                    mode='constant', constant_values=0))
                for x in self.hpqa_batch])

        self.data_word = torch.stack([
                torch.LongTensor(np.pad(x.data_word, 
                    pad_width=[(0, self.max_num_sent-x.sent_count),(0, self.max_sent_len-x.max_sent_len)], mode='constant', constant_values=PAD_WORD))
                for x in self.hpqa_batch])

        self.data_word_ex = torch.stack([
                torch.LongTensor(np.pad(x.data_word_extended, 
                    pad_width=[(0, self.max_num_sent-x.sent_count),(0, self.max_sent_len-x.max_sent_len)], mode='constant', constant_values=PAD_WORD))
                for x in self.hpqa_batch])

        self.data_char = torch.stack([
                torch.LongTensor(np.pad(x.data_char, 
                    pad_width=[(0, self.max_num_sent-x.sent_count), (0, self.max_sent_len-x.max_sent_len), (0, self.max_word_len-x.max_word_len)],
                    mode='constant', constant_values=0))
                for x in self.hpqa_batch])

        self.data_mask = torch.stack([
                torch.LongTensor(np.pad(x.data_word_mask, 
                    pad_width=[(0, self.max_num_sent-x.sent_count), (0, self.max_sent_len-x.max_sent_len)], 
                    mode='constant', constant_values=0))
                for x in self.hpqa_batch])

        self.gold_word = torch.stack([
                torch.LongTensor(np.pad(x.gold_word, 
                    pad_width=[(0, self.max_gold_len - len(x.gold_word))], 
                    mode='constant', constant_values=PAD_WORD))
                for x in self.hpqa_batch])

        self.gold_word_extended = torch.stack([
                torch.LongTensor(np.pad(x.gold_word_extended, 
                    pad_width=[(0, self.max_gold_len - len(x.gold_word_extended))], 
                    mode='constant', constant_values=PAD_WORD))
                for x in self.hpqa_batch])

        self.gold_mask = torch.stack([
                torch.FloatTensor(np.pad(x.gold_mask, 
                    pad_width=[(0, self.max_gold_len - len(x.gold_mask))], 
                    mode='constant', constant_values=0))
                for x in self.hpqa_batch])

        self.gold_mask_unk = torch.stack([
                torch.FloatTensor(np.pad(x.gold_mask_unk, 
                    pad_width=[(0, self.max_gold_len - len(x.gold_mask_unk))], 
                    mode='constant', constant_values=0))
                for x in self.hpqa_batch])

        self.length_s = torch.stack([
                torch.from_numpy(np.pad(x.length_s,
                    pad_width=[(0, self.max_num_sent - len(x.length_s))], 
                    mode='constant', constant_values=0))
                for x in self.hpqa_batch])

        self.length_q = torch.stack([
                torch.from_numpy(np.pad(x.length_q,
                    pad_width=[(0, self.max_length_q-x.question_len)], 
                    mode='constant', constant_values=0))
                for x in self.hpqa_batch])

        self.adjm = torch.stack([
                torch.FloatTensor(np.pad(x.adjm,
                    pad_width=[(0,self.max_num_sent-x.adjm.size()[0]),(0,self.max_num_sent-x.adjm.size()[0])],
                    mode='constant', constant_values=0))
                for x in self.hpqa_batch])

        self.all_to_all = torch.stack([
                torch.FloatTensor(np.pad(x.all_to_all,
                    pad_width=[(0,self.max_num_sent-x.sent_count),(0,self.max_num_sent-x.sent_count)],
                    mode='constant', constant_values=0))
                for x in self.hpqa_batch])

        self.sent_sim = torch.stack([
                torch.FloatTensor(np.pad(x.sent_sim,
                    pad_width=[(0, self.max_num_sent - len(x.sent_sim))], 
                    mode='constant', constant_values=0))
                for x in self.hpqa_batch])

        self.sent_attn_mask = torch.stack([
                torch.FloatTensor(np.pad(x.sent_attn_mask,
                    pad_width=[(0, self.max_num_sent - len(x.sent_attn_mask))], 
                    mode='constant', constant_values=0))
                for x in self.hpqa_batch])

        self.gold = list(map(lambda topic: topic.gold, self.hpqa_batch))
        self.name = list(map(lambda topic: topic.name, self.hpqa_batch))
        self.word_embd = None
    '''
    def __del__(self):
        del self.sent_attn_mask
        del self.sent_sim
        del self.all_to_all
        del self.adjm
        del self.data_mask
        del self.data_mask_no_q
    '''
    def embed(self, embeddings):
        batch_size, docu_len, sent_len = self.data_word.size()
        self.word_embd = embeddings(self.data_word.view(batch_size*docu_len, -1))
        self.word_embd = self.word_embd.view(batch_size * docu_len, sent_len, -1)


class DataSet():

    def __init__(self, json_file, args):
        self.char_vocab_size = CHAR_VOCAB_SIZE
        self.args = args
        self.hpqa_file = json_file
        self.hpqa_file_list = os.listdir(self.hpqa_file)

        self.hpqa_files = list()
        self.hpqa_file_list.sort()
        for i in range(len(self.hpqa_file_list)):
            self.hpqa_files.append(hpqa_short_hand(self.hpqa_file_list[i], i))

        self.data = list()
        self.train = self.data
        self.valid = list()
        self.test = list()

        self.order = None

    # tested + works
    def split_train_valid_test(self, ratio):
        n = len(self.hpqa_files)
        if not self.order:
            self.order = list(range(n))
            random.shuffle(self.order)
        split_list = list()
        train_size = int(n*ratio[0])
        valid_size = int(n*ratio[1])

        self.train = [self.hpqa_files[i] for i in self.order[:train_size]]
        self.valid = [self.hpqa_files[i] for i in self.order[train_size:train_size+valid_size]]
        self.test = [self.hpqa_files[i] for i in self.order[train_size+valid_size:]]

        obj = {}
        obj['train'] = [t.id  for t in self.train]
        obj['valid'] = [t.id  for t in self.valid]
        obj['test']  = [t.id  for t in self.test]

        assert len(obj['train']) != 0
        assert len(obj['valid']) != 0
        assert len(obj['test']) != 0

        split_list.append(obj)

        with open('hotpotqa_data/hotpotqa_split.json', 'w') as open_file:
            json.dump(split_list, open_file)

    def split_dataset(self, filename, _id):
        f = open(filename)
        split_list = json.load(f)
        f.close()
        obj = split_list[_id]
        uid2topic = {t.id: t for t in self.hpqa_files}
        self.train = [uid2topic[uid] for uid in obj['train']]
        self.valid = [uid2topic[uid] for uid in obj['valid']]
        self.test  = [uid2topic[uid] for uid in obj['test']]

    def get_training_item(self, index, delta = 1):
        to_process = read_hpqa_files(self.hpqa_file, 
        self.train[self.args.batch * index: self.args.batch * (index + delta)],
        self.args)
        return to_process

    def get_test_item(self, index, delta = 1):
        to_process = read_hpqa_files(self.hpqa_file, 
        self.test[self.args.batch * index: self.args.batch * (index + delta)],
        self.args)
        return to_process

    def get_eval_item(self, index, delta = 1):
        to_process = read_hpqa_files(self.hpqa_file, 
        self.valid[self.args.batch * index: self.args.batch * (index + delta)],
        self.args)
        return to_process

def read_hotpotqa_data(vocab_file, json_file, args):
    load_word_vocab(vocab_file)
    return DataSet(json_file, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--batch', type=int, default=4, help='mini-batch size')
    parser.add_argument('--data_path', type=str, default=None, help='data path')
    parser.add_argument('--max_sentence_len', default=32, type=int, help = 'Maximum gold length to generate')
    parser.add_argument('--max_word_len', default=24, type=int, help = 'Maximum gold length to generate')
    parser.add_argument('--max_question_len', default=128, type=int, help = 'Maximum gold length to generate')
    parser.add_argument("--tokenization_type", default='spacy', choices=['vanilla','spacy'], type=str, help="tokenization type")

    args = parser.parse_args()
    print(f"Loaing vocab")

    if args.tokenization_type == "spacy":
        txt_file = args.data_path + "preprocessed_sp/hotpotqa_vocab.txt"
        json_file = args.data_path + "preprocessed_sp/train_text/"
        split_file = args.data_path + "hotpotqa_split.json"
    else:
        txt_file = args.data_path + "preprocessed/hotpotqa_vocab.txt"
        json_file = args.data_path + "preprocessed/train_text/"
        split_file = args.data_path + "hotpotqa_split.json"

    # "hotpotqa_data/preprocessed/hotpotqa_vocab.txt"
    load_word_vocab(txt_file)
    # "hotpotqa_data/preprocessed/train_text/", "hotpotqa_data/preprocessed/train_adjm/"
    dataset = DataSet(json_file, args)
    # dataset.split_train_valid_test([0.7, 0.2, 0.1]) # run once to shuffle the dataset
    dataset.split_dataset(split_file, 0)
    # dataset is ready now

    print(f"INFO: word count: {word_count}")
    # lets try getting a sample
    cpu_embedding = nn.Embedding(word_count, 300)
    cpu_embedding.weight.data.copy_(embedding)
    cpu_embedding.weight.requires_grad = False
    train_item = dataset.get_training_item(0, cpu_embedding, 2)
    print(f"train_item.gold_word: {train_item[0].gold}")
    just_ids = [x.id for x in dataset.train]
    print(f"dataset.train: {just_ids[:10]}")
    just_names = [x.name for x in dataset.train]
    print(f"dataset.train: {just_names[:10]}")

