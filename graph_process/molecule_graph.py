# molecule_graph.py
from rdkit import Chem
import numpy as np
import csv
import pickle
import os
from tqdm import tqdm
from torch_geometric.data import Data
from graph_process.features import atom_to_feature_vector, bond_to_feature_vector
import torch
# HELPERS/UTILS
"""
Helper class implementations based on: https://github.com/myprecioushh/ZeroBind
"""
def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def smiles2graph(smiles_string, inchy_key):
    file = "./datasets/pickle_molecule_graph/" + inchy_key + ".pkl"
    if os.path.exists(file):
        with open(file, "rb") as f:
            try:
                data = pickle.load(f)
            except Exception as e:
                print(file, e)
                exit()
        return data

    mol = Chem.MolFromSmiles(smiles_string)
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features_list, dtype=np.int64)
    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    x = convert_to_single_emb(torch.tensor(x))
    # 构造 Data 对象（统一使用 torch_geometric.data.Data）
    data = Data(x=x,
                edge_index=torch.tensor(edge_index),
                edge_attr=torch.tensor(edge_attr))
    data.num_nodes = x.shape[0]

    with open("./datasets/pickle_molecule_graph/" + inchy_key + ".pkl", "wb") as f:
        pickle.dump(data, f)
    return data


def get_molecules_graph():
    molecule_graphs = {}
    with open('datasets/drug_smiles.csv', 'r') as f:
        reader = csv.reader(f)
        for m, row in tqdm(enumerate(reader), desc="molecule graph processing"):
            data = smiles2graph(row[1], row[0])
            molecule_graphs[row[0]] = data
    f.close()
    return molecule_graphs