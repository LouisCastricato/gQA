import time
from model_code.gnn import *
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from model_code.gnn_bridge import GNN_Bridge
from model_code.loss_functions import *
from model_code.data_opinosis import *

import sys
sys.path.append("..")
from utils.state import State
from model_code.gQA_Span import gQA_Span

def _aggregate_avg_pooling(input, text_mask):
    '''
    Apply mean pooling over time
    input: batch_size * max_text_len * rnn_size
    output: batch_size * rnn_size
    '''
    # zero out padded tokens
    batch_size, max_text_len, _ = input.size()
    #print(text_mask.size())
    # idxes = torch.arange(0, int(torch.max(text_len)), out=torch.LongTensor(torch.max(text_len))).unsqueeze(0).cuda()
    # text_mask = Variable((idxes < text_len.unsqueeze(1)).unsqueeze(2).float())
    input = input * text_mask.detach().unsqueeze(2).float()

    out = torch.sum(input, dim=1)
    text_len = text_mask.float().sum(dim=1)
    text_len = text_len.unsqueeze(1).expand_as(out)
    text_len = text_len + text_len.eq(0).float()
    out = out.div(text_len)
    return out

class gSUM_Span(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, args, 
    pretrained_emb=None, sample_size = 3, n_mult = None, d_mult = None):
        super(gSUM_Span, self).__init__()

        #Having a whole copy of gQA span makes transfering easier
        self.base = gQA_Span(word_vocab_size, char_vocab_size, 
        args, pretrained_emb, sample_size, n_mult, d_mult)
        self.ext_gcn = GNN_Bridge(args, 1, 1, sample_size=sample_size, over_gdim=3, steps=8)
        
        self.answer_classifier = nn.Sequential(
            nn.Linear(int(self.hidden_size * 3), self.hidden_size)
            nn.Dropout(args.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1))

        self.answer_criterion = torch.nn.BCEWithLogitsLoss(reduce=None, reduction='none')
        self.softmax = nn.Softmax(dim=2)
        self.drop = nn.Dropout(p=args.dropout)
        self.args = args


    #Changes sample size for testing and evaluation
    def change_sample_size(self, new_ss):
        self.gnn_bridge.sample_size = new_ss
        self.gnn_bridge.sentence_selector.sample_size = new_ss
        self.decoder_lstm.sample_size = new_ss
        self.sample_size = new_ss

    def decode_one_step(self, decoder_input,
    decoded_words, decoder_hidden, bridge_output, coverage, 
    extra_zeros, tf = False, batch_size = 10, PG = True,
    is_coverage = False, gold_word = None, di = 0, oov = None):
        return None

    def forward(self, data_word, data_char, data_mask, length_s,
                adj, all_to_all, sent_sim = None, sent_attn_mask = None, 
                ext = None, dev = None):
        
        
        #Base model
        batch_size, docu_len, sent_len, word_len = data_char.size()
        h_word, h_sent, h_gcn, h_question_sent_sp = \
         self.base(data_word, data_char,
            data_mask, length_s, adj, all_to_all, sent_sim, sent_attn_mask,
            dev=dev, gSUM_Span=True)

        #The extractive summarization embedding
        ext_embd = torch.cat([h_sent, h_gcn, h_question_sent_sp], dim=-1)
        gcn_input = State(outputs=[h_word, ext_embd], mask=data_mask,
        other=[adj, all_to_all, length_s, dev, None, [batch_size, docu_len, sent_len]])
        gcn_output = self.ext_gcn(gcn_input, None, sent_sim=None, span=True, return_attn=True)[1][-1]

        gcn_output = gcn_output.view(batch_size * docu_len, -1)

        answer = self.answer_classifier(gcn_output).view(-1)

        mask = sent_attn_mask.view(-1).eq(0).data
        answer.data.masked_fill_(mask, -float('inf'))
        answer = answer.view(batch_size,-1)


        loss = self.answer_criterion(answer, ext.view(batch_size, -1)).unsqueeze(2)
        loss = _aggregate_avg_pooling(loss, sent_attn_mask).squeeze()

        print(h_sent.size())
        print(h_gcn.size())
        print(h_question_sent_sp.size())

        return loss



        