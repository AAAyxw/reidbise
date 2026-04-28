import os
import sys
import pdb
import json
import numpy as np
import faiss 
from Algorithm.libs.logger.log import get_logger
log_info = get_logger(__name__)
#向量检索
class SearchEngine(object):
    def __init__(self, base_feat_lists, base_idx_lists, dims=1024):
        if len(base_idx_lists) > 0:
            base_feat_lists = np.array(base_feat_lists)
            self._index = faiss.IndexFlatL2(dims)
            self._index.add(base_feat_lists)
            self._register_labels = base_idx_lists
            log_info.info("Faiss search engine load succeed!!! The dims is {}. Total num is {}".format(dims, len(base_idx_lists)))
        else:
            log_info.info("No feat register.Total num is {}".format(len(base_idx_lists)))

    def search(self, query_feat, top_k = 10):
        if len(self._register_labels) == 0:
            return [], []
        else:
            dist_list, idx_list = self._index.search(np.array([query_feat]), top_k)
            label_idx = [self._register_labels[sort_e_idx] for sort_e_idx in idx_list[0]]
            return  label_idx, dist_list
    
    def rerank(self):
        ## Todo
        pass
    
