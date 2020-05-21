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

class gSUM(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, args, 
    pretrained_emb=None, sample_size = 3, n_mult = None, d_mult = None):
        super(gSUM, self).__init__()
        self.sample_size = sample_size
        self.model = args.model.split('-')
        self.word_vocab_size = word_vocab_size
        self.hidden_size = args.d_word_embed
        self.char_cnn = Character_CNN(char_vocab_size, args.d_char_embed, args.n_filter, args.filter_sizes)
        self.word_emb = nn.Embedding(word_vocab_size, self.hidden_size)
        if pretrained_emb is not None:
            self.word_emb.weight.data.copy_(pretrained_emb)
            self.word_emb.weight.requires_grad = False
        d_in = args.n_filter * len(args.filter_sizes) + args.d_word_embed

        #Standard GRU encoders and decoders
        self.encoder_lstm = Word_RNN(d_in, self.hidden_size * args.gdim, cell_type="LSTM", pool=args.pool)
        self.gnn_bridge = GNN_Bridge(args, n_mult, d_mult, sample_size)
        self.query_context_hx = nn.Linear(self.hidden_size * args.gdim * args.d_mult, 
        self.hidden_size * args.gdim * (args.d_mult - 1))
        self.query_context_cx = nn.Linear(self.hidden_size * args.gdim * args.d_mult, 
        self.hidden_size * args.gdim * (args.d_mult - 1))


        if d_mult is None:
            self.decoder_lstm = AttnDecoder_RNN(self.hidden_size * args.gdim, word_vocab_size, 
            sample_size=self.sample_size, dmult=max(args.d_mult, 1), gdim=args.gdim)
        else:
            self.decoder_lstm = AttnDecoder_RNN(self.hidden_size, word_vocab_size, 
            sample_size=self.sample_size, dmult=max(d_mult, 1), gdim=args.gdim)
        self.softmax = nn.Softmax(dim=2)
        self.drop = nn.Dropout(p=args.dropout)
        self.args = args
        self.AF_criterion = torch.nn.BCEWithLogitsLoss()

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
        #Input to the decoder.
        dc_inp = self.word_emb(decoder_input.view(1, -1)).view(1, batch_size, -1)

        #Run the decoder for a single step
        decoder_output, decoder_hidden, attn, coverage = self.decoder_lstm(
            dc_inp, decoder_hidden, bridge_output.outputs[0],
            coverage, mask=bridge_output.mask, pointer_generation=PG,
            extra_zeros=extra_zeros, extended=bridge_output.outputs[1], 
            is_coverage=is_coverage, sample_attn=bridge_output.other)
        #Beam decoding
        if decoded_words is None:
            return decoder_output, decoder_hidden, attn, coverage
        if not tf:
            #Append all of the decoder outputs. @Future Louis: Notice that the indexes are recorded here
            #Word lookup happens in the test function. This cuts down on dictionary overhead significantly
            for i in range(batch_size):
                #Get the most likely word
                _, topi_dec = decoder_output[i].topk(1)
                #If the word is OOV, set the input to UNK, but append the correct word to decoded words.
                #Maybe we can circumvent this? WIP
                if topi_dec.item() >= WORD_VOCAB_SIZE:
                    decoder_input[i] = UNK_WORD
                    indx = topi_dec.item() - WORD_VOCAB_SIZE
                    if not indx >= len(oov[i]):
                        decoded_words[i][di] = topi_dec.item()
                    else:
                        decoded_words[i][di] = UNK_WORD

                #Not OOV? Continue like normal
                else:
                    decoder_input[i] = topi_dec.detach()
                    decoded_words[i][di] = topi_dec.item() 
        else:
            decoder_input = gold_word[di+1]
        return decoder_output, decoder_hidden, attn, coverage, decoded_words, decoder_input

    def forward(self, data_word, data_char, data_mask, length_s,
                adj, all_to_all, sent_sim = None, sent_attn_mask = None, 
                gold_word = None, gold_mask_unk = None, gold_mask = None, 
                cuda = True, dev = None, test = False,
                teacher_forcing_ratio=0.35, lambda_coeff=1.0, gold_len=None, 
                PG=False, oov=None, data_ex=None, gold_ex=None, extra_zeros=None, eps = 1e-12,
                decode = True, stage_two = False):

        # SEQUENCE ENCODING

        batch_size, docu_len, sent_len, word_len = data_char.size()

        #Length of the entire document
        word_embd = self.word_emb(data_word.view(batch_size*docu_len, -1))
        word_embd = word_embd.view(batch_size * docu_len, sent_len, -1)


        data_mask = data_mask.view(batch_size * docu_len, -1)
        length_s = length_s.data.view(batch_size * docu_len)

        '''
        h_word, h_feature, h_sent = sent_encoder(word_embd, length_s)
        '''

        #Length of the gold standard
        if gold_len is None:
            batch_size, gold_len = gold_word.size()
        else:
            batch_size, _ = gold_word.size()
        start = time.time()
        #CharCNNs. Nothing special
        char_emb = self.char_cnn(data_char.long().view(-1, word_len), data_mask.view(-1))

        char_emb = char_emb.view(batch_size*docu_len, sent_len, -1)
        #Combine GloVe with the result from the charCNN
        h_word = torch.cat((char_emb, word_embd), dim=2)


        ### encoding the question (Not needed for gSUM)
        h_question_sent = torch.zeros(batch_size, self.hidden_size).to(dev)


        #These transposes makes the decoding code much easier to read although they are not necesary.
        # print(f"gold_word: {gold_word}")
        # print(f"gold_mask: {gold_mask}")
        # print(f"gold_mask_unk: {gold_mask_unk}")
        # print(f"gold_ex: {gold_ex}")
        # print(f"gold_len: {gold_len}")
        # quit()
        gold_word = torch.t(gold_word)
        gold_mask = torch.t(gold_mask)
        gold_mask_unk = torch.t(gold_mask_unk)
        gold_ex = torch.t(gold_ex)


        start = time.time()

        h_word = self.drop(h_word)
        h_word, h_sent = self.encoder_lstm(h_word, length_s, data_mask, None, device=dev)
        h_sent = h_sent.view(batch_size, docu_len, -1)

        #BRIDGE
        bridge_input = State(outputs=[h_word, h_sent], mask=data_mask,
        other=[adj, all_to_all, length_s, dev, data_ex, [batch_size, docu_len, sent_len]])

        bridge_output = self.gnn_bridge(bridge_input, h_question_sent, sent_sim=None)

        
        gcn_and_question = torch.cat((bridge_output.context[0], h_question_sent.unsqueeze(0)), dim=2)
        bridget_output_context_hx = F.relu(self.query_context_hx(gcn_and_question))

        gcn_and_question = torch.cat((bridge_output.context[1], h_question_sent.unsqueeze(0)), dim=2)
        bridget_output_context_cx = F.relu(self.query_context_cx(gcn_and_question))

        bridge_output.context = (bridget_output_context_hx, bridget_output_context_cx)
        
        #This array is used for testing/evaluation.
        decoded_words = torch.full((batch_size, min(gold_len - 1, MAX_SENTENCE_LEN - 1)), PAD_WORD)
        
        #SOS is the begining of the sentence
        decoder_input = torch.tensor([SOS] * batch_size, device = dev, requires_grad = False)
        decoder_input = decoder_input.view(batch_size)        

        #See paper on pointer generation for context and coverage vectors
        coverage = torch.zeros(batch_size, MAX_SENTENCE_LEN * self.sample_size, requires_grad = True).to(dev)

        # Without teacher forcing: use its own predictions as the next input
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        use_teacher_forcing = False if test else use_teacher_forcing

        #PG's code explicitly states that coverage is only enabled in the last few epochs.
        #This is accounted for here.
        is_coverage = (lambda_coeff > 0)

        #Pre-initialization of variables
        step_losses = []
        loss = 0.
        if PG:
            oov_size = 0
            #Some error handling, if the run_model function set extra_zeros = None, there is no OOV
            if len(extra_zeros.size()) == 2:
                _, oov_size = extra_zeros.size()        
        
        # SEQUENCE DECODING. 
        # print(f"bridge_output: {bridge_output.context.shape}")
        # print(f"h_question_word: {torch.cat((h_question_sent.unsqueeze(0), h_question_sent.unsqueeze(0)), dim=2).shape}")
        # print(f"h_question_sent: {h_question_sent.shape}")

        decoder_hidden = bridge_output.context
        if decode:
            for di in range(min(gold_len - 1, MAX_SENTENCE_LEN - 1)):
                decoder_output, decoder_hidden, attn, \
                coverage, decoded_words, decoder_input \
                = self.decode_one_step(decoder_input, decoded_words, 
                decoder_hidden, bridge_output, coverage, extra_zeros,
                use_teacher_forcing, batch_size, PG=PG, gold_word=gold_word, 
                di=di, oov=oov, is_coverage=is_coverage)

                #LOSS
                step_loss = loss_basic(self.word_vocab_size, loss_func, decoder_output, gold_word, gold_ex, di,
                lambda_coeff, coverage, attn, gold_mask_unk, gold_mask, batch_size, dev, PG, eps=eps)
                
                step_losses.append(step_loss)
            #Average our losses across all steps
            if loss_func is not None:
                loss = sum_loss(step_losses)
                return [loss, 0], decoded_words, None
            else:
                return [0, 0], decoded_words, None
        else:
            return decoder_input, decoded_words, decoder_hidden, bridge_output, \
                use_teacher_forcing, batch_size, coverage