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
import index_embed

COL_ID = "id"
COL_FILE = "path_file"
path = "./faiss_meta.csv"

class FaissObject():
    def __init__(self):
        self.df_metadata = pd.read_csv(path)
        self.index = faiss.read_index(DENSE_INDEX_PATH)

    def search(self, text_embed):
        text_embed = text_embed.detach().cpu().numpy()
        _,doc_id = self.index.search(text_embed, topk)
        doc_id = doc_id[0]
        raw_reference_list = _get_reference_list(doc_id)
        return raw_reference_list
        
    def _get_reference_list(self, doc_id):
        cnt = 0
        raw_reference_list = []
        for i, id in enumerate(doc_id):
            path_file = self._get_json_path(id)
            raw_reference = self._read_json(path_file)
            raw_reference_list.append(raw_reference)

            cnt += 1
            if (cnt == topk):
                break
        return raw_reference_list

    def _get_json_path(self, id):
        mask = self.df_metadata[COL_ID] == str(id)
        path_file = self.df_metadata[mask].iloc[0][COL_FILE]
        return path_file
    
    def _read_json(self, file_path):
        with open(file_path, "r", encoding = "utf-8") as f:
            doc = json.load(f)
        return doc
    
model, tokenizer = index_embed.get_model(is_eval=True)
faiss_object = FaissObject()

text_embed = query_embed(model, tokenizer, text)
faiss_object.search(text_embed)
