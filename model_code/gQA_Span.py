import sys
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from model_code.gnn import *
from model_code.gnn_bridge import GNN_Bridge
from model_code.loss_functions import *
from model_code.data_hotpotqa import *
from model_code.attention import BiAttention, MultiHeadAttention

sys.path.append("..")
from utils.state import State

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
class gQA_Span(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, args, pretrained_emb=None, n_mult=None, d_mult=None):
        super(gQA_Span, self).__init__()
        self.args = args
        self.model = args.model.split('-')
        self.word_vocab_size = word_vocab_size
        self.hidden_size = args.d_word_embed
        self.char_cnn = Character_CNN(char_vocab_size, args.d_char_embed, args.n_filter, args.filter_sizes)


        self.word_emb = nn.Embedding(word_vocab_size, self.hidden_size)
        self.pretrained_emb = pretrained_emb
        if pretrained_emb is not None:
            self.word_emb = nn.Embedding.from_pretrained(pretrained_emb, freeze=True)
            self.word_emb.weight.requires_grad = False

        d_in = args.n_filter * len(args.filter_sizes) + args.d_word_embed

        #Standard GRU encoders and decoders
        self.encoder_lstm = Word_RNN(d_in, self.hidden_size * args.gdim, cell_type="LSTM", pool=args.pool)
        self.gnn_bridge = GNN_Bridge(args, n_mult, d_mult, steps=8)
        self.query_context = nn.Linear(self.hidden_size * args.gdim * args.d_mult, 
        self.hidden_size * args.gdim * (args.d_mult - 1))


        #Decoding
        self.decoder_lstm = Word_RNN(self.hidden_size, self.hidden_size, cell_type="LSTM", pool=args.pool)
        #self.sp_gcn = GNN_Bridge(args, n_mult, d_mult, sample_size, steps=8)
        self.answer_lstm = Word_RNN(self.hidden_size * 2, self.hidden_size * 2, cell_type="LSTM", pool=args.pool)
        self.answer_end_lstm = Word_RNN(self.hidden_size * 2, self.hidden_size * 2, cell_type="LSTM", pool=args.pool)

        #Finding answer + supporting sentences
        self.answer_type_gcn = GNN_Bridge(args, n_mult, d_mult, over_gdim=2, steps=8)

        #Reduction
        self.end_word_reduce = nn.Linear(self.hidden_size*4, self.hidden_size*2)

        self.end_sent_reduce = nn.Linear(self.hidden_size*4, self.hidden_size*2)
        self.type_sent_reduce = nn.Linear(self.hidden_size*4, self.hidden_size*2)

        self.answer_start_classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size*2, 1))

        self.answer_end_classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size*2, 1))


        self.sp_linear = nn.Linear(int(self.hidden_size * 2), self.hidden_size)
        self.sp_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 1))

        self.answer_type_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2,3))

        self.drop = nn.Dropout(p=args.dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.sp_criterion = torch.nn.CrossEntropyLoss(reduce=None, reduction='none')
        self.type_criterion = torch.nn.CrossEntropyLoss(reduce=None, reduction='none')
        self.start_criterion = torch.nn.CrossEntropyLoss(reduce=None, reduction='none')
        self.end_criterion = torch.nn.CrossEntropyLoss(reduce=None, reduction='none')


        #Bi-attention and self-attention for the sentence tokens
        self.biattn = BiAttention(self.hidden_size, args.dropout)
        self.biattn_linear = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.ReLU()
        )
        self.biattn_rnn = Word_RNN(self.hidden_size, self.hidden_size, cell_type="LSTM", pool="mean")
        
        self.selfattn = BiAttention(self.hidden_size, args.dropout)
        self.selfattn_linear = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.ReLU()
        )
        self.selfattn_rnn = Word_RNN(self.hidden_size, self.hidden_size, cell_type="LSTM", pool="mean")

    #Changes sample size for testing and evaluation
    def change_sample_size(self, new_ss):
        self.gnn_bridge.sample_size = new_ss
        self.gnn_bridge.sentence_selector.sample_size = new_ss
        self.decoder_lstm.sample_size = new_ss
        self.sample_size = new_ss


    def forward(self, data_word, data_char, data_mask, length_s,
                adj, all_to_all, sent_sim, sent_attn_mask, 
                question_word=None, question_word_mask=None, 
                question_char=None, question_char_mask=None, length_q=None,
                dev = None,  start_pos=None, end_pos=None, 
                a_type=None, answer_mask=None, data_mask_no_q=None, gSUM_Span=False):
        # SEQUENCE ENCODING

        #Length of the entire document
        batch_size, docu_len, sent_len = data_word.size()
        word_embd = self.word_emb(data_word.view(batch_size*docu_len, -1))
        word_embd = word_embd.view(batch_size * docu_len, sent_len, -1)

        batch_size, docu_len, sent_len, word_len = data_char.size()

        start = time.time()
        #CharCNNs. Nothing special
        char_emb = self.char_cnn(data_char.long().view(-1, word_len), data_mask.view(-1))

        char_emb = char_emb.view(batch_size*docu_len, sent_len, -1)
        #Combine GloVe with the result from the charCNN
        h_word = torch.cat((char_emb, word_embd), dim=2)

        data_mask = data_mask.view(batch_size * docu_len, -1)
        length_s = length_s.data.view(batch_size * docu_len)

        ### encoding the question
        # question word embedding
        if not gSUM_Span:
            batch_size, question_len = question_word.size()
            question_word_embd = self.word_emb(question_word.view(batch_size, -1))

            # question character embedding
            batch_size, question_len, question_word_len = question_char.size()        
            question_char_emb = self.char_cnn(question_char.long().view(-1, question_word_len), question_word_mask.view(-1))
            question_char_emb = question_char_emb.view(batch_size, question_len, -1)

            # question char + word embedding
            h_question_word = torch.cat((question_char_emb, question_word_embd), dim=2)
            question_word_mask = question_word_mask.view(batch_size, -1)
            length_q = length_q.data.view(batch_size)
            h_question_word = self.drop(h_question_word)

            h_question_word, h_question_sent = self.encoder_lstm(h_question_word, length_q, \
                question_word_mask, None, device=dev)
            h_question_sent = h_question_sent.view(batch_size, -1)
            
        h_word = self.drop(h_word)
        h_word = self.encoder_lstm(h_word, length_s, \
            data_mask, None, device=dev, pool_skip=True)

        if not gSUM_Span:

            #Bi-attention
            output = self.biattn(h_word.view(batch_size, -1, self.hidden_size), \
                h_question_word, question_word_mask)
            output = self.biattn_linear(output)
            output = output.view(batch_size * docu_len, sent_len, -1)

            output_t = self.biattn_rnn(output, length_s, data_mask, None, device=dev, pool_skip=True)

            if args.global_self_attn:
                #Self attention. Sentences are considered jointly
                output_t = self.selfattn(output_t.view(batch_size, docu_len*sent_len, -1), \
                    output_t.view(batch_size, docu_len*sent_len, -1), data_mask.view(batch_size, docu_len*sent_len))
                output_t = self.selfattn_linear(output_t.view(batch_size * docu_len, sent_len, -1))
            else:
                #Self attention. Sentences are considered disjointly
                output_t = self.selfattn(output_t, output_t, data_mask)
                output_t = self.selfattn_linear(output_t)
                
            #Combine outputs
            output = output.view(h_word.size())
            output_t = output_t.view(h_word.size())
            output = output + output_t

            #Resize for re-encode
            output = output.view(batch_size*docu_len, sent_len, -1)

            #Re-encode the sentences
            h_word, h_sent = self.selfattn_rnn(output, length_s, data_mask, None, device=dev)
            h_sent = h_sent.view(batch_size, docu_len, -1)

        #Question is set to empty in this case
        if gSUM_Span:
            h_question_sent = torch.zeros((batch_size, self.hidden_size)).to(dev)

        #BRIDGE
        bridge_input = State(outputs=[h_word, h_sent], mask=data_mask,
        other=[adj, all_to_all, length_s, dev, None, [batch_size, docu_len, sent_len]])


        bridge_output = self.gnn_bridge(bridge_input, h_question_sent, sent_sim=sent_sim, span=True, return_attn=True)


        gcn_and_question = torch.cat((bridge_output[0], h_question_sent), dim=1)
        h_gcn = F.relu(self.query_context(gcn_and_question))
        h_gcn_output = bridge_output[1][-1]

        #SEQUENCE DECODING
        h_word, h_sent = self.decoder_lstm(h_word, length_s, data_mask, \
            h_gcn_output.view(-1, self.hidden_size), device=dev)

        #CLASSIFIER

        #Predict supporting sentences. This is where the document summary is introduced
        #gcn_input = State(outputs=[h_word, h_sent.view(batch_size, docu_len, -1)], mask=data_mask,
        #other=[adj, all_to_all, length_s, dev, None, [batch_size, docu_len, sent_len]])
        #bridge_output = self.sp_gcn(gcn_input, h_question_sent, sent_sim=None, span=True, return_attn=True)
        #sp_gcn_output = bridge_output[1][-1]

        h_sent = h_sent.view(h_gcn_output.size())
        #sp_gcn_output = sp_gcn_output.view(h_gcn_output.size())
        #h_question_sent_sp = h_question_sent.unsqueeze(1).expand(batch_size, docu_len, self.hidden_size).contigous()
        #h_question_sent_sp = h_question_sent_sp.view(h_sent.size())


        sp_output = self.sp_linear(torch.cat([h_sent, h_gcn_output], dim=-1))
        sp_logits = self.sp_classifier(F.relu(self.drop(sp_output)))

        #Fill for crossentropy
        mask = sent_attn_mask.view(-1).eq(0).data
        sp_logits.view(-1).data.masked_fill_(mask, -float('inf'))

        #Append the output from sp_output
        h_sent = torch.cat([h_sent, sp_output], dim=-1)
        sp_output = sp_output.view(-1, self.hidden_size).unsqueeze(-2).expand(h_word.size()).contiguous()
        h_word = torch.cat([h_word, sp_output], dim=-1)


        if gSUM_Span:
            #word embedding, sentence embedding, document embedding, supporting sentence embedding
            return h_word, h_sent, h_gcn, h_question_sent_sp


        #ANSWER START/END
        h_word_start, h_start = self.answer_lstm(h_word, length_s, data_mask, h_sent.view(-1, h_sent.size(-1)), device=dev)
        start_logits = self.answer_start_classifier(h_word_start)
        h_word = self.end_word_reduce(torch.cat([h_word, h_word_start], dim=-1))
        h_sent = self.end_sent_reduce(torch.cat([h_sent, h_start.view(batch_size, -1,  h_sent.size(-1))], dim=-1))

        h_word_end, h_end = self.answer_end_lstm(h_word, length_s, data_mask, h_sent.view(-1, h_sent.size(-1)), device=dev)
        end_logits = self.answer_end_classifier(h_word_end)

        
        #TYPE
        h_sent = self.type_sent_reduce(torch.cat([h_sent, h_end.view(batch_size, -1,  h_sent.size(-1))], dim=-1))

        gcn_input = State(outputs=[h_word, h_sent], mask=data_mask,
        other=[adj, all_to_all, length_s, dev, None, [batch_size, docu_len, sent_len]])
        h_answer_type = self.answer_type_gcn(gcn_input, h_question_sent, sent_sim=None, span=True, return_attn=True)[0]

        type_logits = self.answer_type_classifier(h_answer_type)
        
        #COMPUTE LOSS
        loss = 0

        sp_aux = Variable(sp_logits.data.new(sp_logits.size(0), sp_logits.size(1), 1).zero_())
        sp_loss = torch.cat([sp_aux, sp_logits], dim=-1).contiguous().view(-1, 2)
        sent_sim = sent_sim.view(-1)
        sp_loss = self.sp_criterion(sp_loss, sent_sim.type(torch.LongTensor).to(dev))

        sp_loss = _aggregate_avg_pooling(sp_loss.view(batch_size, -1, 1), sent_attn_mask).squeeze()


        #Mask out pad words
        mask = data_mask_no_q.view(-1).eq(0).data
        start_logits.view(-1).data.masked_fill_(mask, -float('inf'))
        end_logits.view(-1).data.masked_fill_(mask, -float('inf'))

        start_logits = start_logits.view(batch_size,-1)
        end_logits = end_logits.view(batch_size,-1)
        
        answer_loss = (self.start_criterion(start_logits, start_pos) + \
            self.end_criterion(end_logits, end_pos)) * answer_mask

        if a_type.size()[0] != 1:
            type_loss = self.type_criterion(type_logits.squeeze(), a_type)
        else:
            type_loss = self.type_criterion(type_logits, a_type) 

        loss = type_loss
        loss += answer_loss
        loss += sp_loss * args.sp_lambda
        
        for i, end in enumerate(self.end_criterion(end_logits, end_pos)):
            if end == float('inf'):
                print(i)
                print(end_pos[i])
                print(data_mask[i][end_pos[i]])
                return "Error", i


        return loss, torch.sigmoid(sp_logits.squeeze()), start_logits.squeeze(), \
            end_logits.squeeze(), self.softmax(type_logits.squeeze())