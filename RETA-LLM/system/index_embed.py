import json
import faiss
import torch
import numpy as np
from config import *
from model_response import generate_response
import os
import pandas as pd

import sys
sys.path.append("..")
from dense_model import *
from transformers import AutoConfig, AutoTokenizer, AutoModel
from adaptertransformers.src.transformers import PretrainedConfig

from system.config import is_cuda

device = torch.device("cuda") if is_cuda else torch.device("cpu")

def get_model(is_eval=True):
    config = PretrainedConfig.from_pretrained(DAM_NAME)
    config.similarity_metric, config.pooling = "ip", "average"
    tokenizer = AutoTokenizer.from_pretrained(DAM_NAME, config=config)
    model = BertDense.from_pretrained(DAM_NAME, config=config)
    adapter_name = model.load_adapter(REM_URL)
    model.set_active_adapters(adapter_name)
    if is_eval: 
        model.eval()
    model.to(device)
    return model, tokenizer

def tokenized_text(tokenizer, text):
    tokens = tokenizer(text, 
                       padding='max_length', 
                       truncation=True, 
                       max_length=512, 
                       return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    return input_ids, attention_mask

def query_embed(model, tokenizer, text):
    input_ids, attention_mask = tokenized_text(tokenizer, text)
    text_embed = model(input_ids=input_ids, 
                        attention_mask=attention_mask)
    return text_embed

