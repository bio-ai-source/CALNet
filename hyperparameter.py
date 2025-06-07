# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/7/05
@author: Qichang Zhao
"""
import torch
from datetime import datetime
class hyperparameter():
    def __init__(self):
        self.current_time = datetime.now()
        self.type = 'normal'
        self.DATASET = 'DrugBank'

        self.K_Fold = 5
        self.epochs = 50
        self.Learning_rate = 0.01
        self.batch = 16
        self.seed_cross = 42
        self.weight_decay = 5e-4
        self.device = 'cuda'
        self.num_heads=3
        self.dropout = 0.5
        self.num_layers = 4
        self.l_c = 6
        self.l_d = 2048
        self.mlp_layer_output = 3

