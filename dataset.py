from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pathlib import Path

class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        
        self.seq_len = seq_len
        self.ds = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        self.sos_tok = torch.tensor([tokenizer_src.token_to_id("<SOS>")], dtype = torch.int64)
        self.eos_tok = torch.tensor([tokenizer_src.token_to_id("<EOS>")], dtype = torch.int64)
        self.pad_tok = torch.tensor([tokenizer_src.token_to_id("<PAD>")], dtype = torch.int64)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence too long')
        
        encoder_input = torch.cat(
            [
                self.sos_tok,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_tok,
                torch.tensor([self.pad_tok] * enc_num_padding_tokens, dtype = torch.int64)
            ]
        )
        
        decoder_input = torch.cat(
            [
                self.sos_tok,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                # self.eos_tok,
                torch.tensor([self.pad_tok] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )
        
        label = torch.cat(
            [
                # self.sos_tok,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_tok,
                torch.tensor([self.pad_tok] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )
        
        assert encoder_input.shape[0] == self.seq_len
        assert decoder_input.shape[0] == self.seq_len
        assert label.shape[0] == self.seq_len
        
        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.pad_tok).unsqueeze(0).unsqueeze(0).int(), #(1, 1, seq_len)
            'decoder_mask': (decoder_input != self.pad_tok).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.shape[0]), #(1, 1, seq_len) & (1, seq_len, seq_len)
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }



def causal_mask(dim):

    all_ones = torch.ones(1, dim, dim)
    mask = torch.tril(all_ones, diagonal = 1).type(torch.int)
    #tril method sets all the values above the diagonal as zero. Returning lower triangular matrix.
    
    return mask == 1 #everything with 1 becomes true and else becomes false.

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["<UNK>", "<PAD>", "<SOS>", "<EOS>"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer


def get_dataset(config):
    dataset = load_dataset("cfilt/iitb-english-hindi")
    
    tokenizer_src = get_tokenizer(config, dataset["train"], config['lang_src'])
    tokenizer_tgt = get_tokenizer(config, dataset["train"], config['lang_tgt'])
    
    train_dataset = BilingualDataset(dataset["train"], tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_dataset = BilingualDataset(dataset["validation"], tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    max_len_src = 0
    max_len_tgt = 0

    for item in dataset['train']:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    train_loader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)

    return train_loader, val_loader, tokenizer_src, tokenizer_tgt
