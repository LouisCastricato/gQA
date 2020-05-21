import argparse


import sys
sys.path.append("..")
import torch

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--epochs', type=int, default=35, metavar='N', help='number of epochs to train (default: 40)')
parser.add_argument('--start', type=int, default=0, help='epoch to start on. Effects lr, tfr, and lambda.')
parser.add_argument('--batch', type=int, default=12, help='mini-batch size')
parser.add_argument('--file_limit', type=int, default=400, help='Number of files to use')
parser.add_argument('--cache', type=int, default=10, help='Number of batches to preload')
parser.add_argument('--lr', type=float, default=4e-4, metavar='LR', help='learning rate (default: 1e-3 Determined empirically)')
parser.add_argument('--wd', type=float, default=1e-6, help='weight decay (default: 1e-6)')
parser.add_argument('--d_mult', type=int, default=2, help='Size multiplier of the decoder.')
parser.add_argument('--n_mult', type=int, default=1, help='Number of GCN layers.')
parser.add_argument('--cuda', action='store_true', default=True, help='enable CUDA training')
parser.add_argument('--cuda_device', type=int, default=0, help='Default CUDA device')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 123123)')
parser.add_argument('--d_char_embed', type=int, default=64, help='character embedding dimension')
parser.add_argument('--d_word_embed', type=int, default=200, help='word embedding dimension')
parser.add_argument('--filter_sizes', type=str, default='2,3,4', help='character cnn filter size')
parser.add_argument('--n_filter', type=int, default=64, help='character cnn filter number')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
parser.add_argument('--model', type=str, default='lstm-ggcn-lstm', choices=['lstm', 'lstm-lstm', \
'lstm-gcn-lstm', 'lstm-gat-lstm', \
'lstm-posgat-lstm', 'lstm-ggcn-lstm'], help='model')

parser.add_argument('--save_path', type=str, default='models/output', help='output name')
parser.add_argument('--data_path', type=str, default='sum_data/', help='data path')
parser.add_argument('--task', type=str, default='opinosis', choices=['opinosis','hotpotqa','squad'])
parser.add_argument('--test', action='store_true', default=False, help='test mode')
parser.add_argument('--validate', action='store_true', default=False, help='validate mode')
parser.add_argument('--load', action='store_true', default=False, help='Load prior model')
parser.add_argument('--save', action='store_true', default=False, help='Save model')
parser.add_argument('--split', action='store_true', default=False, help = 'Regenerate the train/val/test split')
parser.add_argument('--debug', default = False, help = 'Enable debug')
parser.add_argument('--test_indx', default=2, type=int, help = 'Index to perform testing on')
parser.add_argument('--gold_len', default=32, type=int, help = 'Maximum gold length to generate')
parser.add_argument('--checkpoint_load',    action='store_true', default=False, help = 'Enables checkpoint loading.')
parser.add_argument('--train_qa',    action='store_true', default=False, help = 'Enables QA training')
parser.add_argument('--postgcn_attn',    action='store_true', default=False, help = 'Uses the post GCN embeddings for the sentence attn mechanism')
parser.add_argument('--condition_on_q',    action='store_true', default=False, help = 'Condition the sentence embeddings on the question')


#EXPERIMENTAL
parser.add_argument('--pool', type=str, default='mean', choices=['attn', 'mean', 'max', 'last'], help='Pooling type')
parser.add_argument('--gdim', default=1, type=int, help = 'An integer that increases the hidden size of our LSTM.')
parser.add_argument('--on_colab', action='store_true', default=False, help = 'Are we running the code on colab? Saves best model to google drive.')
parser.add_argument('--TPU', action='store_true', default=False, help = 'Are we using a TPU?')
parser.add_argument('--span', action='store_true', default=False, help = 'Compute spans instead of using an LSTM.')
parser.add_argument('--finetune_embd', action='store_true', default=False, help = 'Finetunes the GloVe embeddings')
parser.add_argument('--sp_lambda', type=float, default=5.0, help='Coefficient to bias sp loss by.')

parser.add_argument('--max_sentence_len', default=20, type=int, help = 'Maximum gold length to generate')
parser.add_argument('--max_word_len', default=24, type=int, help = 'Maximum gold length to generate')
parser.add_argument('--max_question_len', default=128, type=int, help = 'Maximum gold length to generate')
parser.add_argument("--tokenization_type", default='spacy', choices=['vanilla','spacy'], type=str, help="tokenization type")
parser.add_argument("--encoder_type", default='vanilla', choices=['vanilla','pgen'], type=str, help="tokenization type")
parser.add_argument("--global_self_attn",  action='store_true', default=False, help = 'Determines whether attention spans between sentences.')

args = parser.parse_args()

args.filter_sizes = [int(x) for x in args.filter_sizes.split(',')]


WORD_VOCAB_SIZE = 50000
word_count = WORD_VOCAB_SIZE
d_output = word_count
args.d_graph = [args.d_word_embed * args.gdim] * args.n_mult
print(args)
result_obj = {}

#Changes current cuda device. Useful since multiple experiments are ran in parallel
if args.TPU:
    import torch_xla
    import torch_xla.core.xla_model as xm
    cuda_device = xm.xla_device()
else:
    if args.cuda:
        torch.cuda.set_device(int(args.cuda_device))

    cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

