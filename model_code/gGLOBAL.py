import time
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model_code.gnn import *
from model_code.gnn_bridge import GNN_Bridge
from model_code.loss_functions import *
from model_code.data_opinosis import *

sys.path.append("..")
from seq2seq_pytorch.seq2seq.models.modules.state import State

class gGLOBAL(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, args, pretrained_emb=None, sample_size = 3, train_qa = False):
        super(gGLOBAL, self).__init__()
        self.sample_size = sample_size
        self.model = args.model.split('-')
        self.train_gqa = args.train_gqa
        self.word_vocab_size = word_vocab_size
        self.hidden_size = args.d_word_embed
        self.char_cnn = Character_CNN(char_vocab_size, args.d_char_embed, args.n_filter, args.filter_sizes)
        self.word_emb = nn.Embedding(word_vocab_size, self.hidden_size)
        if pretrained_emb is not None:
            self.word_emb.weight.data.copy_(pretrained_emb)
            self.word_emb.weight.requires_grad = False
        d_in = args.n_filter * len(args.filter_sizes) + args.d_word_embed

        #Standard GRU encoders and decoders
        self.encoder_lstm = Word_RNN(d_in, self.hidden_size, cell_type="LSTM")
        self.gnn_bridge = GNN_Bridge(args)
        self.query_context = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.decoder_lstm = AttnDecoder_RNN(self.hidden_size, word_vocab_size, 
        sample_size=self.sample_size, dmult=min(args.d_mult, 1))

        self.softmax = nn.Softmax(dim=2)
        self.drop = nn.Dropout(p=args.dropout)
        self.args = args

    #Changes sample size for testing and evaluation
    def change_sample_size(self, new_ss):
        self.gnn_bridge.sample_size = new_ss
        self.gnn_bridge.sentence_selector.sample_size = new_ss
        self.decoder_lstm.sample_size = new_ss
        self.sample_size = new_ss
    def forward(self, loss_func, data_word, data_char, data_mask, length_s,
                adj, all_to_all, sent_attn_neg, 
                gold_word = None, gold_mask_unk = None, gold_mask = None, 
                question = None, cuda = True, dev = None, test = False,
                teacher_forcing_ratio=0.35, lambda_coeff=1.0, gold_len=None, 
                PG=False, oov=None, data_ex=None, gold_ex=None, extra_zeros=None, eps = 1e-12):
        
        # SEQUENCE ENCODING

        #Length of the entire document
        batch_size, docu_len, sent_len = data_word.size()
        word_embd = self.word_emb(data_word.view(batch_size*docu_len, -1))
        word_embd = word_embd.view(batch_size * docu_len, sent_len, -1)

        batch_size, docu_len, sent_len, word_len = data_char.size()
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

        data_mask = data_mask.view(batch_size * docu_len, -1)
        length_s = length_s.data.view(batch_size * docu_len)

        #These transposes makes the decoding code much easier to read although they are not necesary.
        gold_word = torch.t(gold_word)
        gold_mask = torch.t(gold_mask)
        gold_mask_unk = torch.t(gold_mask_unk)
        gold_ex = torch.t(gold_ex)

        start = time.time()
       
        h_word = self.drop(h_word)
        h_word, h_sent = self.encoder_lstm(h_word, length_s, data_mask, None)
        h_sent = h_sent.view(batch_size, docu_len, -1)

        '''
        question_word = None, question_word_mask = None, 
        question_ex = None, question_char = None, question_char_mask = None, length_q = None,
        '''
        if self.train_gqa is not None:
            question_word = question.word
            question_word_mask = question.word_mask
            question_ex = question.ex
            question_char = question.char
            question_char_mask = question.char_mask
            length_q = question.length_q

            ### encoding the question
            # question word embedding
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

            h_question_word, h_question_sent = self.encoder_lstm(h_question_word, length_q, question_word_mask, None)
            h_question_sent = h_question_sent.view(batch_size, -1)

        
        #BRIDGE
        bridge_input = State(outputs=[h_word, h_sent], mask=data_mask,
        other=[adj, all_to_all, length_s, dev, data_ex, [batch_size, docu_len, sent_len]])

        bridge_output = self.gnn_bridge(bridge_input)
        if self.train_gqa:
            gcn_and_question = torch.cat((bridge_output.context, h_question_sent.unsqueeze(0)), dim=2)
            bridge_output.context = self.query_context(gcn_and_question)

        #This array is used for testing/evaluation.
        decoded_words = torch.full((batch_size, min(gold_len - 1, MAX_SENTENCE_LEN - 1)), PAD_WORD)
        
        #SOS is the begining of the sentence
        decoder_input = torch.tensor([SOS] * batch_size, device = dev, requires_grad = False)
        decoder_input = decoder_input.view(batch_size)        

        #See paper on pointer generation for context and coverage vectors
        context = torch.zeros(batch_size, self.hidden_size, requires_grad = True).to(dev)
        coverage = torch.zeros(batch_size, MAX_SENTENCE_LEN * self.sample_size, requires_grad = True).to(dev)

        # Without teacher forcing: use its own predictions as the next input
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

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
        if self.train_gqa:
            decoder_hidden = h_question_sent.unsqueeze(0)
        else:
            decoder_hidden = bridge_output.context

        # First branch is no teacher forcing, or test.
        if (not use_teacher_forcing) or test:
            for di in range(min(gold_len - 1, MAX_SENTENCE_LEN - 1)):
                #Input to the decoder.
                dc_inp = self.word_emb(decoder_input.view(1, -1)).view(1, batch_size, -1)

                #Run the decoder for a single step
                decoder_output, decoder_hidden, attn, context, coverage = self.decoder_lstm(
                    dc_inp, decoder_hidden, bridge_output.outputs[0], context,
                    coverage, mask=bridge_output.mask, pointer_generation=PG,
                    extra_zeros=extra_zeros, extended=bridge_output.outputs[1], is_coverage=is_coverage)

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

                #LOSS
                step_loss = loss_basic(self.word_vocab_size, loss_func, decoder_output, gold_word, gold_ex, di,
                lambda_coeff, coverage, attn, gold_mask_unk, gold_mask, batch_size, dev, PG, eps=eps)
                
                step_losses.append(step_loss)

        #This branch is teacher forcing.
        else:
            for di in range(min(gold_len - 1, MAX_SENTENCE_LEN - 1)):

                #Input to the decoder.
                dc_inp = self.word_emb(decoder_input.view(1, -1)).view(1, batch_size, -1)

                #Run the decoder for a single step
                decoder_output, decoder_hidden, attn, context, coverage = self.decoder_lstm(
                    dc_inp, decoder_hidden, bridge_output.outputs[0], context,
                    coverage, mask=bridge_output.mask, pointer_generation=PG,
                    extra_zeros=extra_zeros, extended=bridge_output.outputs[1], is_coverage=is_coverage)

                #LOSS
                step_loss = loss_basic(self.word_vocab_size, loss_func, decoder_output, gold_word, gold_ex, di,
                    lambda_coeff, coverage, attn, gold_mask_unk, gold_mask, batch_size, dev, PG, eps=eps)
                
                step_losses.append(step_loss)

                #Teacher forcing
                decoder_input = gold_word[di+1]
        #Average our losses across all steps
        loss = sum_loss(step_losses)

        return [loss, attn_forcing_loss(sent_attn_neg, bridge_output.attention_score, self.args.AF)], decoded_words, None