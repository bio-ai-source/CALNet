import random
import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd
from torch_geometric.loader import DataLoader
from collections import defaultdict
from torch.optim import Optimizer
import numpy as np


def show_result(DATASET1, type, DATASET2,  Accuracy_List,Precision_List,Recall_List,F1_score_List,AUC_List,AUPR_List):
    Accuracy_mean = np.mean(Accuracy_List)
    Precision_mean = np.mean(Precision_List)
    Recall_mean = np.mean(Recall_List)
    F1_score_mean = np.mean(F1_score_List)
    AUC_mean = np.mean(AUC_List)
    PRC_mean = np.mean(AUPR_List)
    print("The results on {} of {}  :".format(DATASET1,DATASET2))
    with open(f'results/{DATASET2}/{type}/results.txt', 'a') as f:
        f.write('Accuracy:{:.4f}'.format(Accuracy_mean) + '\n')
        f.write('Precision:{:.4f}'.format(Precision_mean) + '\n')
        f.write('Recall:{:.4f}'.format(Recall_mean) + '\n')
        f.write('F1_score:{:.4f}'.format(F1_score_mean) + '\n')
        f.write('AUC:{:.4f}'.format(AUC_mean) + '\n')
        f.write('PRC:{:.4f}'.format(PRC_mean) + '\n')
        f.write(f'Accuracy: {Accuracy_List}' + '\n')
        f.write(f'Precision: {Precision_List}' + '\n')
        f.write(f'Recall: {Recall_List}' + '\n')
        f.write(f'AUC_List: {AUC_List}'  + '\n')
        f.write(f'AUPR: {AUPR_List}' + '\n')
        f.write(f'F1: {F1_score_List}' + '\n')
        f.write('\n')
    print('Accuracy(std):{:.4f}'.format(Accuracy_mean))
    print('Precision(std):{:.4f}'.format(Precision_mean))
    print('Recall(std):{:.4f}'.format(Recall_mean))
    print('F1_score(std):{:.4f}'.format(F1_score_mean))
    print('AUC(std):{:.4f}'.format(AUC_mean))
    print('PRC(std):{:.4f}'.format(PRC_mean))


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k  # How often to perform the slow parameter update
        self.alpha = alpha  # Smoothing factor
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss
    def zero_grad(self):
        self.optimizer.zero_grad()

def get_name():
    drug_name = np.loadtxt('datasets/d.txt', dtype=str, delimiter=' ')
    protein_name = np.loadtxt('datasets/p.txt', dtype=str, delimiter=' ')
    drug_map = {idx : line for idx, line in enumerate(drug_name)}
    protein_map = {idx : line for idx, line in enumerate(protein_name)}
    return drug_map, protein_map

def load_txt_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            drug_idx, protein_idx, label = map(int, line.split())
            data.append((drug_idx, protein_idx, label))
    return data

def get_kfold_data(i, datasets, k):
    fold_size = len(datasets) // k
    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:]
        trainset = datasets[0:val_start]
    return trainset, validset

def get_data(dataname, datapath):
    dataset = []
    with open(datapath + dataname + '.txt', 'r') as f:
        data_list = f.read().strip().split('\n')
        for data in data_list:
            if dataname == 'DrugBank':
                parts = data.split()
            drug_name, protein_name, label = parts[0], parts[1], parts[-1]
            dataset.append((drug_name, protein_name, label))
    random.shuffle(dataset)
    return dataset


def get_embedding():
    df = pd.read_csv('datasets/d_features.csv', header=None)
    drug_embedding, protein_embedding = {}, {}
    for _, row in df.iterrows():
        drug_name = row[1]
        feature_vector = row[2:].values  #
        drug_embedding[drug_name] = feature_vector
    df = pd.read_csv('datasets/p_features.csv', header=None)
    for _, row in df.iterrows():
        protein_name = row[0]
        feature_vector = row[1:].values
        protein_embedding[protein_name] = feature_vector
    return drug_embedding, protein_embedding

def DI_DataLoader(data, molecule_graph, protein_graph, molecule_embedding, protein_embedding):
    data_samples = []
    for sample in data:
        drug_name, protein_name, label = sample
        dg = molecule_graph[drug_name]
        dt = torch.tensor(np.array(molecule_embedding[drug_name], dtype=np.float32), dtype=torch.float)
        label = torch.tensor(int(label))

        data_samples.append(Data(
            x = dg.x,
            edge_index= dg.edge_index,
            edge_attr= dg.edge_attr,
            dt = dt,
            drug_name = drug_name,
            protein_name = protein_name,
            label = label
            )
        )
    return data_samples

def PI_DataLoader(data, molecule_graph, protein_graph, molecule_embedding, protein_embedding):
    data_samples = []
    for sample in data:
        drug_name, protein_name, label = sample
        label = torch.tensor(int(label))
        pg = protein_graph[protein_name]
        pt = torch.tensor(np.array(protein_embedding[protein_name], dtype=np.float32), dtype=torch.float)

        data_samples.append(Data(
            x = pg.x,
            edge_index= pg.edge_index,
            pt = pt,
            drug_name = drug_name,
            protein_name = protein_name,
            label = label
            )
        )
    return data_samples

def get_dataloader(train_data, valid_data, test_data,  molecule_graph, protein_graph, molecule_text_embedding, protein_text_embedding, batch):
    train_drug_samples = DI_DataLoader(data=train_data, molecule_graph=molecule_graph, protein_graph=protein_graph,
                                       molecule_embedding=molecule_text_embedding,
                                       protein_embedding=protein_text_embedding)
    train_protein_samples = PI_DataLoader(data=train_data, molecule_graph=molecule_graph, protein_graph=protein_graph,
                                          molecule_embedding=molecule_text_embedding,
                                          protein_embedding=protein_text_embedding)
    valid_drug_samples = DI_DataLoader(data=valid_data, molecule_graph=molecule_graph, protein_graph=protein_graph,
                                       molecule_embedding=molecule_text_embedding,
                                       protein_embedding=protein_text_embedding)
    valid_protein_samples = PI_DataLoader(data=valid_data, molecule_graph=molecule_graph, protein_graph=protein_graph,
                                          molecule_embedding=molecule_text_embedding,
                                          protein_embedding=protein_text_embedding)
    test_drug_samples = DI_DataLoader(data=test_data, molecule_graph=molecule_graph, protein_graph=protein_graph,
                                      molecule_embedding=molecule_text_embedding,
                                      protein_embedding=protein_text_embedding)
    test_protein_samples = PI_DataLoader(data=test_data, molecule_graph=molecule_graph, protein_graph=protein_graph,
                                         molecule_embedding=molecule_text_embedding,
                                         protein_embedding=protein_text_embedding)

    train_drug_loader = DataLoader(train_drug_samples, batch_size=batch, shuffle=False)
    train_protein_loader = DataLoader(train_protein_samples, batch_size=batch, shuffle=False)
    valid_drug_loader = DataLoader(valid_drug_samples, batch_size=batch, shuffle=False)
    valid_protein_loader = DataLoader(valid_protein_samples, batch_size=batch, shuffle=False)
    test_drug_loader = DataLoader(test_drug_samples, batch_size=batch, shuffle=False)
    test_protein_loader = DataLoader(test_protein_samples, batch_size=batch, shuffle=False)
    return train_drug_loader, train_protein_loader, valid_drug_loader, valid_protein_loader, test_drug_loader, test_protein_loader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False