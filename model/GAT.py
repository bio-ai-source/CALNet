from torch.nn.functional import dropout
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch import nn, einsum
import torch



class GAT(nn.Module):
    def __init__(self, molecule_in_channels=512, molecule_out_channels=1024, molecule_num_heads=3, molecule_dropout=0.5,
                 protein_in_channels=1280, protein_out_channels=1280, protein_num_heads=3, protein_dropout=0.5,
                 molecule_layer_norm=True, protein_layer_norm=True):
        super(GAT, self).__init__()

        self.layer_weights = nn.Parameter(torch.randn(20))

        self.num_layers = 4
        # 分子
        self.molecule_atomencoder = nn.Embedding(512 * 9 + 1, molecule_in_channels, padding_idx=0)
        self.num_heads = molecule_num_heads
        self.dropout = dropout
        self.layer_norm = molecule_layer_norm
        self.molecule_gat_layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.molecule_gat_layers.append(
                    GATConv(molecule_in_channels, molecule_out_channels, heads=molecule_num_heads,
                            concat=False))  # concat=False
            else:
                self.molecule_gat_layers.append(
                    GATConv(molecule_out_channels, molecule_out_channels, heads=molecule_num_heads,
                            concat=False))  # concat=False
        self.molecule_dropout_layer = nn.Dropout(p=molecule_dropout)
        if molecule_layer_norm:
            self.molecule_layer_norms = nn.ModuleList(
                [nn.LayerNorm(molecule_out_channels) for _ in range(self.num_layers)])  # out_channels
        else:
            self.molecule_layer_norms = None
        self.molecule_pool = global_mean_pool
        # 蛋白质
        self.protein_num_heads = protein_num_heads
        self.protein_dropout = protein_dropout
        self.protein_layer_norm = protein_layer_norm
        self.protein_gat_layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.protein_gat_layers.append(
                    GATConv(protein_in_channels, protein_out_channels, heads=protein_num_heads,
                            concat=False))  # concat=False
            else:
                self.protein_gat_layers.append(
                    GATConv(protein_out_channels, protein_out_channels, heads=protein_num_heads,
                            concat=False))  # concat=False
        self.protein_dropout_layer = nn.Dropout(p=protein_dropout)
        if protein_layer_norm:
            self.protein_layer_norms = nn.ModuleList(
                [nn.LayerNorm(protein_out_channels) for _ in range(self.num_layers)])  # out_channels
        else:
            self.protein_layer_norms = None
        self.protein_pool = global_mean_pool





    def forward(self, drug_batch, protein_batch):
        molecule_x, molecule_edge_index, molecule_batch = self.molecule_atomencoder(
            drug_batch.x.long()), drug_batch.edge_index, drug_batch.batch
        molecule_x = torch.mean(molecule_x, dim=-2)
        molecule_xx = self.molecule_pool(molecule_x, molecule_batch).unsqueeze(1)

        protein_x, protein_edge_index, protein_batch = protein_batch.x, protein_batch.edge_index, protein_batch.batch
        protein_xx = self.protein_pool(protein_x, protein_batch).unsqueeze(1)

        molecule_features = molecule_xx
        protein_features = protein_xx

        for i in range(self.num_layers):
            molecule_x = self.molecule_gat_layers[i](molecule_x, molecule_edge_index)
            molecule_x = self.molecule_dropout_layer(molecule_x)
            if self.molecule_layer_norms is not None:
                molecule_x = self.molecule_layer_norms[i](molecule_x)
            protein_x = self.protein_gat_layers[i](protein_x, protein_edge_index)
            protein_x = self.protein_dropout_layer(protein_x)
            if self.protein_layer_norms is not None:
                protein_x = self.protein_layer_norms[i](protein_x)
            x = self.molecule_pool(molecule_x, molecule_batch).unsqueeze(1)
            molecule_features = torch.cat((molecule_features, x), dim = 1)
            y = self.protein_pool(protein_x, protein_batch).unsqueeze(1)
            protein_features = torch.cat((protein_features, y), dim = 1)
        return molecule_features, protein_features

