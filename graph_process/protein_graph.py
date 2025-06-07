# protein_graph.py
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from functools import partial
from graphein.protein.edges.distance import add_distance_threshold, add_peptide_bonds
import esm
import networkx as nx
import os
import torch
import pandas
import warnings
import pickle
from torch_geometric.data import Data
from tqdm import tqdm

import torch
# HELPERS/UTILS
"""
Helper class implementations based on: https://github.com/myprecioushh/ZeroBind
"""
pandas.set_option('mode.chained_assignment', None)
protein_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
protein_model.eval()
new_edge_funcs = {"edge_construction_functions": [partial(add_distance_threshold, long_interaction_threshold=0, threshold=8)]}
config = ProteinGraphConfig(**new_edge_funcs)

def adj2table(adj):
    edge_index = [[], []]
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if int(adj[i][j]) != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.tensor(edge_index, dtype=torch.long)

def pretrain_protein(data):
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    results = protein_model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    feat = token_representations.squeeze(0)[1:len(data[0][1]) + 1]
    return feat

def graph_node(pdb_ID, seq):
    if len(seq) > 1022:
        seq_feat = []
        for i in range(len(seq) // 1022):
            data = [(pdb_ID, seq[i * 1022:(i + 1) * 1022])]
            seq_feat.append(pretrain_protein(data))
        remainder = len(seq) % 1022
        if remainder > 0:
            data = [(pdb_ID, seq[-remainder:])]
            seq_feat.append(pretrain_protein(data))
        seq_feat = torch.cat(seq_feat, dim=0)
    else:
        data = [(pdb_ID, seq)]
        seq_feat = pretrain_protein(data)
    return seq_feat

def protein_graph(protein_path, pdb_ID):
    file = "./datasets/pickle_protein_graph/" + pdb_ID + ".pkl"
    if os.path.exists(file):
        with open(file, "rb") as f:
            data = pickle.load(f)
        return data  # 直接返回 Data 对象
    pdb_file = str(protein_path) + str(pdb_ID) + ".pdb"
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    else:
        print('exist')
    g = construct_graph(config=config, path=str(protein_path) + str(pdb_ID) + ".pdb")
    A = nx.to_numpy_array(g, nonedge=0, weight='distance')
    edge_index = adj2table(A)
    seq = ""
    for key in g.graph.keys():
        if key.startswith("sequence_"):
            seq += g.graph[key]
    if len(seq) != g.number_of_nodes():
        raise RuntimeError("number of nodes mismatch")
    node_feat = graph_node(pdb_ID, seq)
    # 构造 Data 对象
    data = Data(x=node_feat.detach(), edge_index=edge_index.detach())
    data.num_nodes = node_feat.size(0)
    with open("./datasets/pickle_protein_graph/" + pdb_ID + ".pkl", "wb") as f:
        pickle.dump(data, f)
    return data



def get_proteins_graph():
    protein_graphs = {}
    with open('datasets/p_pdb.txt', 'r') as f:
        for m, line in enumerate(tqdm((f), desc="protein graph processing")):
            row = line.strip().split()
            protein_id = row[0]
            # 将蛋白质ID和图数据作为元组添加到列表中
            data = protein_graph('./datasets/alphafold_structures/', protein_id)
            protein_graphs[protein_id] = data
    return protein_graphs
