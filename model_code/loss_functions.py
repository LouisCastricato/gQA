from model_code.gnn import *
import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from model_code.gnn_bridge import GNN_Bridge
from model_code.data_opinosis import *

def loss_basic(word_vocab_size, loss_func, decoder_output, gold_word, gold_ex, di,
            lambda_coeff, coverage, attn, gold_mask_unk, gold_mask, batch_size, dev,
            pointer_generation = False, eps = 1e-12):
        gold_probs = 0
        step_loss = 0

        if not pointer_generation:
            #OOV is UNK, so this is fine
            step_loss = loss_func(decoder_output[:, 0:word_vocab_size], gold_word[di+1])
            step_loss = step_loss + lambda_coeff * torch.sum(torch.min(coverage, attn), dim=1)
            step_loss = step_loss * gold_mask[di+1]
        else:
            '''
            #Anything that should no longer be UNK needs to have its log probability minimized
            decoder_distribution = F.softmax(decoder_output, dim=1)[:, 0:word_vocab_size]
            unk_prob = (1 - decoder_distribution.contiguous().view(-1))
            step_loss = -torch.log(unk_prob + torch.full(unk_prob.size(), eps).to(dev))
            step_loss = torch.sum(step_loss.view(batch_size, -1), 1) * gold_mask_unk[di+1]
            '''
            #Collect the probs for the gold extension
            gold_probs = decoder_output[range(batch_size), gold_ex[di+1]]

            #Construct an appropriately sized epsilon tensor. Then move to GPU. 
            step_loss_secondary = -torch.log(gold_probs + torch.full(gold_probs.size(), eps).to(dev)) 
            #step_loss_secondary += lambda_coeff * torch.sum(torch.min(coverage, attn), dim=1)

            #Mask out all padded elements
            step_loss_secondary = (step_loss_secondary * gold_mask[di+1])
            #Account for new loss
            step_loss = step_loss + step_loss_secondary

        return step_loss


def sum_loss(step_losses):
    return torch.sum(torch.stack(step_losses, 1), 1)

def attn_forcing_loss(sent_sim, sent_attn_mask, sent_attn, AF=False, criterion=None):
    if AF:
        sent_sim = sent_sim.view(-1)
        sent_attn_mask = sent_attn_mask.view(-1)
        sent_attn = sent_attn.view(-1)

        attn_f_loss = torch.sum(criterion(sent_attn, sent_sim) * sent_attn_mask)

        return attn_f_loss
    return 0
    