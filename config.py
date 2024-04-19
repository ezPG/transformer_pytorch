from __future__ import print_function

import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from sklearn.metrics import average_precision_score

def get_config():
    return {
        "seed": 1,
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 1e-3,
        "seq_len": 500,
        "embed_size": 512,
        "lang_src": "en",
        "lang_tgt": "hi",
        "save_path": "checkpoints",
        "model_basename": "translator_model",
        "tokenizer_file": "tokenizer_{0}.json",
        "exp_name": "run/model",
        "preload": None,
        "local_rank": 0
    }

def get_model_file_path(config, epoch):
    model_folder = config['save_path']
    model_basename = config['model_basename']
    model_name = f"{model_basename}-{epoch}.pt"
    
    return str(Path('.')/ model_folder / model_name)

def set_seed(seed):
    logging.info(f'=======> Using Fixed Random Seed: {seed} <========')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False  # set to False for final report
    
    else:
        torch.use_deterministic_algorithms(True, warn_only=True)