from utils.exec_util import *
from utils.config import *
from model_code.data_hotpotqa import *
import signal
import time
import gc
from utils.GGNN import *

class TimeOutException(Exception):
   pass
 
def alarm_handler(signum, frame):
    raise TimeOutException()



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def execute(dataset):
    pool = ThreadPool(processes=1)
    print('split dataset')

    dataset.split_dataset(args.data_path+args.task+'_split.json', 0)
    print('train:', len(dataset.train), 'valid:', len(dataset.valid), 'test:', len(dataset.test))

    dataset.train = dataset.train[:args.file_limit]

    model = gQA_Span(word_vocab_size=word_count, char_vocab_size=CHAR_VOCAB_SIZE, args=args,
                            pretrained_emb=embedding)
    
    model = model.to(cuda_device)

    obj = None
    start_pos = args.start
    #For checkpoints
    index_master = 1
    itt_loss = 0
    scheduler_itt = 0
    if args.load:
        if args.checkpoint_load:
            try:
                print("Loading checkpoint.")
                obj = torch.load(args.save_path+'_checkpoint.model', map_location=lambda storage, loc:storage)
                model.load_state_dict(obj['model'])
                model.to(cuda_device)
                if 'index' in obj:
                    index_master = obj['index']
                if 'loss' in obj:
                    itt_loss = obj['loss']

                print("Checkpoint loaded at batch: " + str(index_master))
            except:
                print("Unable to load checkpoint.")
                pass
        else:
            print("Loading model.")
            obj = torch.load(args.save_path+'.model', map_location=lambda storage, loc:storage)
            model.load_state_dict(obj['model'])
            model.to(cuda_device)
            print('Model loaded.')
    if args.validate:
        print("Current Validation EM: " + \
            str(validate_qa(dataset, run_model, model, pool, args, samples=14)))
        return False
    if args.test:
        print("Current Test Output: ")
        test_qa(dataset, run_model, model, pool, args)
        return False
    
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, amsgrad=True, weight_decay=args.wd)

    if args.load:
        optimizer.load_state_dict(obj['optimizer'])
        start_pos = max(int(obj['start']),  0)
        #print(start_pos)

    #We want the scheduler to decrease every other cycle
    num_batches = args.file_limit/args.batch

    #Warmup, for larger models
    scheduler = StepLR(optimizer, step_size=1, gamma=0.88)
    warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=1,\
        after_scheduler=scheduler)

    if args.load:
        check=False
        print("Loading schedulers.")
        if 'scheduler' in obj:
            scheduler.load_state_dict(obj['scheduler'])
            check=True
        if 'warmup' in obj:
            warmup_scheduler.load_state_dict(obj['warmup'])
            check=True
        if not check:
            print("Unable to load schedulers.")
        
    total_loss = 0

    model.train()

    it_arr = []
    val_arr = []

    #Restore
    if start_pos > 4:
        if args.finetune_embd:
            print("Enabling finetuning of embeddings.")
            model.word_emb.weight.requires_grad = True

    print("Model has: " + str(count_parameters(model)) + " number of params.")

    signal.signal(signal.SIGALRM, alarm_handler)


    # this zero gradient update is needed to avoid a warning message, issue #8.
    optimizer.zero_grad()
    optimizer.step()

    for epoch in range(start_pos, args.epochs):
        
        if epoch == 4:
            if start_pos < 4 and args.finetune_embd:
                model.word_emb = nn.Embedding.from_pretrained(model.pretrained_emb, freeze=False).to(cuda_device)
            if args.finetune_embd:
                #Disable all gradients except the word embeddings
                for param in model.parameters():
                    param.requires_grad = False
                model.word_emb.weight.requires_grad = True
        
        random.shuffle(dataset.train)
        warmup_scheduler.step(epoch)
        print("Currently On Epoch: " + str(epoch))
        print("LR: ")
        print(warmup_scheduler.get_lr())

        num_batches = (args.file_limit * 0.9)/args.batch
        cur_cache = dataset.get_training_item(0, args.cache)
        itt_loss = 0
        itt = 0
        for index in tqdm(range(index_master, int(num_batches), args.cache)):
            #print(index)
            async_result = pool.apply_async(dataset.get_training_item, (index, args.cache))
            #print(cur_cache)
            for batch in cur_cache:
                #Some batches have low data quality
                model.train()
                optimizer.zero_grad()

                signal.alarm(15)
                try:
                    #Time out after 15 seconds
                    loss = run_model(model, batch, False, args.train_qa, True, args.span)[0]

                except TimeOutException as e:
                    print("Batch time out after 15 seconds, skipping batch...")
                    #Skip this batch
                    continue

                signal.alarm(0)

                #Accumulate loss
                itt_loss += loss.sum().data
                #print(itt_loss)
                loss.sum().backward()

                if epoch == 0:
                    warmup_scheduler.step(float(index)/float(num_batches))

                if args.TPU:
                    xm.optimizer_step(optimizer, barrier=True)
                else:
                    optimizer.step()

                #Grad clipping
                clip_grad_norm_(model.encoder_lstm.parameters(), 3.)
                clip_grad_norm_(model.decoder_lstm.parameters(), 3.)
                
                clip_grad_norm_(model.answer_lstm.parameters(), 3.)
                clip_grad_norm_(model.answer_end_lstm.parameters(), 3)

                clip_grad_norm_(model.biattn_rnn.parameters(), 3.)
                clip_grad_norm_(model.selfattn_rnn.parameters(), 3.)

                if type(model.answer_type_gcn.gnn_layer[-1]) is GGNN: 
                    clip_grad_norm_(model.gnn_bridge.gnn_layer[-1].propagator.parameters(), 10.)
                    #clip_grad_norm_(model.sp_gcn.gnn_layer[-1].propagator.parameters(), 10.)
                    clip_grad_norm_(model.answer_type_gcn.gnn_layer[-1].propagator.parameters(), 10.)


            #Try to fetch the next cache, if it fails we stored a backup
            backup = cur_cache
            try:
                cur_cache = async_result.get(5.0)
                del backup
            except Exception as e: 
                #If there was an issue with this batch, just load the next batch and continue
                print(e)
                print("Exception while training! Oh no!")
                cur_cache = backup
                continue
            
            #Save a checkpoint
            if itt % 15 == 0 and args.save:
                obj = {'args':args, 'model':model.state_dict(), 'optimizer':optimizer.state_dict(), \
                    'start': epoch, 'index' : index, \
                        'scheduler' : scheduler.state_dict(), 'warmup' : warmup_scheduler.state_dict()}
                torch.save(obj, args.save_path+'_checkpoint.model')
                if args.on_colab:
                    torch.save(obj, '/content/drive/My Drive/Colab Notebooks/Gits/Graphical-Summarization/' + args.save_path + '_checkpoint.model')
            itt += 1

        it_arr.append(itt_loss)

        total_loss += itt_loss
        print("Total Loss Was: ")
        print(total_loss)
        print("Iteration Loss Was: ")
        print(itt_loss)
        '''
        print("Current Test Output: ")
        test_qa(dataset, run_model, model, pool, args, samples=64)
        '''
        #Save most recent edition, do not check if it is the best
        save(model, optimizer, epoch, None, scheduler=scheduler, warmup=warmup_scheduler)
        
        #Validation
        print("Computing validation accuracy... Please wait.")
        val_em, val_loss, N = \
            validate_qa(dataset, run_model, model, pool, args, samples=1)

        val_arr.append(val_em)
        #scheduler.step(val_em)

        if args.save:
            save(model, optimizer, epoch, val_loss, scheduler=scheduler, warmup=warmup_scheduler)
        
        model.train()
        index_master = 0
        itt = 0

    if args.save:
        save(model, optimizer, epoch, val_arr[-1])

    return True


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    graph = ('gcn' in args.model) or ('gat' in args.model)

    if args.tokenization_type == "spacy":
        txt_file = "hotpotqa_data/preprocessed_sp/hotpotqa_vocab.txt"
        json_file = args.data_path + "preprocessed_sp/train_text/"
    else:
        txt_file = "hotpotqa_data/preprocessed_vn/hotpotqa_vocab.txt"
        json_file = args.data_path + "preprocessed_vn/train_text/"
    dataset = read_hotpotqa_data(txt_file, json_file, args)

    print('Data loaded.')

    if args.split:
        print("Saving split")
        dataset.split_train_valid_test([.9,.05])
        print("Split saved. Begining training")

    if execute(dataset):
        for attr, value in sorted(args.__dict__.items()):
            result_obj[attr] = value
