import string
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model_code.gnn import *
from model_code.data_opinosis import *
from experimental.supporting_sent import supporting_sent
#Determines which sentences we want to sample for the decoder. Essentially performing an
#extractive summary before performing an abstractive one
class Sentence_Selector(nn.Module):
    def _aggregate_avg_pooling(self, input, text_mask):
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

    def __init__(self, args, n_mult = None, d_mult = None, sample_size = None, over_gdim=None):
        super(Sentence_Selector, self).__init__()

        self.model = args.model.split('-')
        if over_gdim is None:
            self.hidden_size = int(args.d_word_embed * args.gdim)
        else:
            self.hidden_size = int(args.d_word_embed * over_gdim)
        self.train_qa = args.train_qa

        hs = self.hidden_size
        d_hs = self.hidden_size + 100

        self.attn_linear_base = nn.Linear(hs, d_hs)
        if args.postgcn_attn:
            self.attn_linear_seq = nn.Linear(hs, d_hs)
        self.attn_linear_query = nn.Linear(args.d_word_embed, d_hs)
        self.sentence_attn = AttnSent(d_hs, max_length_input=1, max_length_output=1)

        self.softmax = nn.Softmax(dim=1)
        self.args = args
    
    def compute_attn_inp(self, gcn_raw, batch_size, docu_len):
        #Combine GCN layers
        base = self.attn_linear_base(gcn_raw[0].view(-1, self.hidden_size))

        '''
        salience = self.attn_linear_all_to_all(gcn_raw[2].view(-1, self.hidden_size))
        '''
        if self.args.postgcn_attn:
            neighbours = self.attn_linear_seq(gcn_raw[-1].view(-1, self.hidden_size))
            #print(neighbours.size())
            #print(base.size())
            output = base + neighbours
        else:
            output = base
        if self.args.train_qa:
            h_dim = gcn_raw[1].size()[-1]
            query = self.attn_linear_query(gcn_raw[1].view(-1, self.args.d_word_embed))
            output += query
        
        return output.view(batch_size, docu_len, -1)
    
    def forward(self, h_word, h_sent, data_mask, length_s,
            dev=None, gcn_layers=None, gcn_raw=None,
            data_ex=None, dimensions=[0, 0, 0], sent_sim=None, span=False, return_attn = False):
        
        #Unpack dimensions
        batch_size, docu_len, sent_len = dimensions


        #resize length
        length_s = length_s.view(batch_size, docu_len)
        decoder_hidden = torch.zeros(batch_size, self.hidden_size)

        #Apply attention to the concat of the sentence and enrichments
        #gcn_layers = gcn_layers.view(batch_size, docu_len, -1)
        doc_lengths_reduced = torch.full((batch_size, 1), 0)
        for i in range(batch_size):
            doc_lengths_reduced[i] = torch.sum(torch.clamp(length_s[i], 0, 1))
        doc_lengths_reduced = doc_lengths_reduced.to(dev).view(-1).int()

        attn_inp = self.compute_attn_inp(gcn_raw, batch_size, docu_len)
        decoder_hidden, sent_attn = self.sentence_attn(attn_inp, dev, lengths=doc_lengths_reduced)

        decoder_hidden = torch.bmm(torch.transpose(gcn_raw[-1], 1, 2), sent_attn.unsqueeze(2)).squeeze(2)
        return decoder_hidden