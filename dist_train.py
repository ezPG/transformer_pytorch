import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
import torchmetrics

# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel
# from torch.distributed import init_process_group, destroy_process_group
from accelerate import Accelerator

from model import build_transformer
from dataset import get_dataset, causal_mask
from config import get_config, get_model_file_path, set_seed

from pathlib import Path
from tqdm import tqdm
import datetime
import logging
import os
import time
import argparse
import warnings


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
                            vocab_src_len, vocab_tgt_len, 
                            config['seq_len'], config['seq_len'], 
                            config['embed_size']
                            )
    return model

def train(config, batch_iterator, model, criterion, optimizer, tokenizer_src, tokenizer_tgt, epoch, global_step):
    model.train()
    
    epoch_loss = 0
    for batch in batch_iterator:
        
        encoder_input = batch['encoder_input']#.to(device)   #(Batch, seq_len)
        decoder_input = batch['decoder_input']#.to(device)   #(Batch, seq_len)
        encoder_mask = batch['encoder_mask']#.to(device)     #(Batch, 1, 1, seq_len)
        decoder_mask = batch['decoder_mask']#.to(device)     #(Batch, 1, seq_len, seq_len)
        
        project_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)    #(Batch, seq_len, tgt_vocab_size)
        
        label = batch['label']#.to(device)   #(Batch, seq_len)
        
        #(Batch, seq_len, tgt_vocab_size) --> (Batch * seq_len, tgt_vocab_size)
        loss = criterion(project_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
        
        # batch_iterator.set_postfix({f'Loss: {loss.item():.3f}'})
        
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    
        epoch_loss += loss.item()
    
    return epoch_loss/config['batch_size']


def greedy_decode(config, model, src, src_mask, tokenizer_src, tokenizer_tgt, max_len):
    
    sos_idx = tokenizer_tgt.token_to_id('<SOS>')
    eos_idx = tokenizer_tgt.token_to_id('<EOS>')
    
    encoder_out = model.encode(src, src_mask)
    decoder_in = torch.empty(1, 1).fill_(sos_idx).type_as(src)#.to(device)
    while True:
        if decoder_in.shape[1] == max_len:
            break
        
        decoder_mask = causal_mask(decoder_in.shape[1]).type_as(src_mask)#.to(device)
        
        decoder_out = model.decode(decoder_in, encoder_out, src_mask, decoder_mask)
        
        prob = model.project(decoder_out[:, -1])
        
        pred = torch.argmax(prob, dim = 1)
        next_token = torch.empty(1, 1).type_as(src).fill_(pred.item())#.to(device)
        
        decoder_in = torch.cat([decoder_in, next_token], dim = 1)
        
        if pred == eos_idx:
            break
        
    return decoder_in.squeeze(0)

def validate(config, val_loader, model, tokenizer_src, tokenizer_tgt, print_msg, global_step, writer, num_examples = 2):
    model.eval()
    count = 0
    
    res = {
        'src_text' : [],
        'expected' : [],
        'predicted': []
    }
    
    max_len = config['seq_len']
    # console_width = 80 #console window size
    
    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80
    
    with torch.no_grad():
        for batch in val_loader:
            count += 1
            encoder_input = batch['encoder_input'].to(device)   #(Batch, seq_len)
            decoder_input = batch['decoder_input'].to(device)   #(Batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)     #(Batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)     #(Batch, 1, seq_len, seq_len)
            src_text = batch['src_text'][0]
            tgt_text = batch['tgt_text'][0]
            
            model_out = greedy_decode(config, model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len)
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            res['src_text'].append(src_text)
            res['expected'].append(tgt_text)
            res['predicted'].append(model_out_text)
            
            print_msg('-'*console_width)
            print_msg(f'SOURCE: {src_text}')
            print_msg(f'TARGET: {tgt_text}')
            print_msg(f'PREDICTED: {model_out_text}')
            
            if count == num_examples:
                print_msg('-'*console_width)
                break
    # Compute the char error rate 
    metric = torchmetrics.CharErrorRate()
    cer = metric(res['predicted'], res['expected'])
    wandb.log({'validation/cer': cer, 'global_step': global_step})

    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(res['predicted'], res['expected'])
    wandb.log({'validation/wer': wer, 'global_step': global_step})

    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(res['predicted'], res['expected'])
    wandb.log({'validation/BLEU': bleu, 'global_step': global_step})
    
    return
    

def main():
    # warnings.filterwarnings('ignore')
    
    
    config = get_config()
    
    global accelerator, device
    
    accelerator = Accelerator()
    device = accelerator.device()
    
    print(f"GPU {config['local_rank']} - Using device: {device}")
    set_seed(config['seed'])
    
    Path(config['save_path']).mkdir(parents=True, exist_ok=True)
    
    
    print(f"GPU {config['local_rank']} - Loading dataset...")
    train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # print(model)
    # model = DistributedDataParallel(model, device_ids=[config['local_rank']])
    criterion = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id('<PAD>'), label_smoothing=0.1).to(device)
    
    init_epoch = 0
    global_step = 0
    num_epochs = config['num_epochs']
    
    decay_epochs = [num_epochs // 3, num_epochs * 2 // 3]
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)

    
    
    if config['preload']:
        model_filename = get_model_file_path(config, config['preload'])
        print(f'Loading saved model from: {model_filename}')
        state = torch.load(model_filename)
        init_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    
    model, optimizer, training_loader, scheduler = accelerator.prepare(
            model, optimizer, training_loader, scheduler
    )
    
    for epoch in range(init_epoch, num_epochs + 1):
        batch_iterator = tqdm(train_loader, desc=f'Epoch: {epoch:02d}')
        logging.info(f'[{epoch} / {num_epochs}] Learning rate: {scheduler.get_last_lr()[0]}')
        loss = train(config, batch_iterator, model, criterion, optimizer, tokenizer_src, tokenizer_tgt, epoch, global_step)
        logging.info(f'[{epoch} / {num_epochs}] Loss: {loss:.3f}')
        
        scheduler.step()
        
        global_step += 1
        
        model_filename = get_model_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        
        validate(config, val_loader, model, tokenizer_src, tokenizer_tgt, lambda msg: batch_iterator.write(msg), global_step)

if __name__ == '__main__':
    main()
