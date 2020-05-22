import string
import time
from model_code.gnn import *
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from model_code.sentence_selector import Sentence_Selector
from model_code.data_hotpotqa import *

import sys
sys.path.append("..")
from utils.state import State
from utils.GGNN import GGNN

def isnan(x):
    return x != x

#Acts as a bridge between the encoder and decoder
class GNN_Bridge(nn.Module):
    def __init__(self, args, n_mult = None, d_mult = None, over_gdim=None, steps=8):
        super(GNN_Bridge, self).__init__()

        self.model = args.model.split('-')
        if over_gdim is None:
            self.hidden_size = int(args.d_word_embed * args.gdim)
        else:
            self.hidden_size = int(args.d_word_embed * over_gdim)
        self.n_mult = args.n_mult
        if n_mult is None:
            self.d_graph = args.d_graph
        else:
            self.d_graph = [args.d_word_embed] * n_mult
            self.n_mult = n_mult
        self.gnn_layer = nn.ModuleList()
        self.over_gdim = over_gdim
        #Selects which sentences to attend over
        self.sentence_selector = Sentence_Selector(args, n_mult, d_mult, over_gdim=over_gdim)

        d_in = self.hidden_size
        i = 1
        for d_out in args.d_graph:
            if 'gcn' in self.model:
                self.gnn_layer.append(GNN_Layer(d_in, d_in, globalnode=False))
            if 'gat' in self.model:
                self.gnn_layer.append(GNN_Att_Layer(n_head=4, d_input=d_in, d_model=d_out, globalnode=False))
            if 'posgat' in self.model:
                self.gnn_layer.append(GNN_Pos_Att_Layer(d_input=d_in, d_model=d_out))
            if 'ggcn' in self.model:
                self.gnn_layer.append(GGNN(state_dim=d_in, annotation_dim=1, n_edge_types=1, n_steps=steps))
            d_in = d_out
            i += 1
            
        self.doc_representation = GNN_Layer(self.hidden_size, self.hidden_size, globalnode=False)

        if args.train_qa:
            self.MAX_SENTENCE_LEN = args.max_sentence_len
        else:
            self.MAX_SENTENCE_LEN = MAX_SENTENCE_LEN
        self.args = args

    def forward(self, context, question=None, sent_sim=None, span = False, return_attn = False):

        #Unpack input
        adj, all_to_all, length_s, dev, data_ex, dimensions = context.other
        batch_size, docu_len, sent_len = dimensions
        h_word, h_sent = context.outputs
        data_mask = context.mask

        h_sent = h_sent.view(batch_size, docu_len, -1)
        h_word = h_word.view(batch_size, docu_len, sent_len, -1)

        h_gcn = None
        data_mask = data_mask.view(batch_size, docu_len, sent_len)
        h_gcn = h_sent
        gcn_layers = []
        #Append the actual sentence embeddings
        gcn_layers.append(h_gcn)

        #Append the propogated sentence embeddings
        if question is not None:
            if len(question.size()) != 3:
                question = question.unsqueeze(1).expand(question.size(0), h_gcn.size(1), question.size(1)).contiguous()
            else:
                _,_,h_dim = question.size()
                question = question.view(batch_size, -1, h_dim)
            gcn_layers.append(question)
        for i in range(len(self.gnn_layer)):
            h_gcn = self.gnn_layer[i](h_gcn, adj)
            gcn_layers.append(h_gcn)

        # gcn_layers: num layeres, batch, max article len, sent dim
        h_gcn = h_gcn.to(dev)
        
        #If we don't do this, selecting the sentence to pass to the decoder becomes much harder
        h_word = h_word.view(batch_size, docu_len, sent_len, -1)
       
        #Pad h_word so that all sentences are the same length. This does not really make a difference,
        #as we never attend to this.
        p2d = (0,0, self.MAX_SENTENCE_LEN - sent_len, 0)
        h_word = F.pad(h_word, p2d, 'constant', 0)

        # ATTENTION MECHANISM
        output = self.sentence_selector(
        h_word, h_sent, data_mask, length_s, dev=dev, gcn_layers=None, gcn_raw = gcn_layers,
        data_ex=data_ex, dimensions=[batch_size, docu_len, sent_len], sent_sim=sent_sim, span=span, return_attn=return_attn)

        return output, gcn_layers