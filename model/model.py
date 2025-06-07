
from torch import nn
import torch
from .GAT import GAT
from .CALNet import CALNet
class DTI(nn.Module):
    def __init__(
            self,
            drug_in_channels,
            drug_out_channels,
            protein_in_channels,
            protein_out_channels,
            num_heads,
            num_layers,
            dropout,
            layer_output,
            l_c,
            l_d
    ):
        super(DTI, self).__init__()

        self.gat = GAT(
            molecule_in_channels=drug_in_channels, molecule_out_channels=drug_out_channels, molecule_num_heads=num_heads, molecule_dropout=dropout,
            protein_in_channels=protein_in_channels, protein_out_channels=protein_out_channels, protein_num_heads=num_heads, protein_dropout=dropout,
        )

        self.fusion = CALNet(
            n_modalities=4,
            channel_dims=[1024, 1280, 3072, 3072],
            num_spatial_axes=[1, 1, 0, 0],
            out_dims=2,  # 输出的维度
            fourier_encode_data=False,
            l_c=l_c,
            l_d=l_d,
        )
        self.layer_output = layer_output

        self.W_out = nn.ModuleList([nn.Linear(
            2048, 1024),
            nn.Linear(1024, 512)])
        # mlp=3
        self.W_interaction = nn.Linear(512, 2)
    def forward(self, drug_batch, protein_batch):
        device = 'cuda'
        drug_graph, protein_graph = self.gat(drug_batch, protein_batch)
        drug_graph = drug_graph.to(device)
        protein_graph = protein_graph.to(device)

        batch = drug_graph.size(0)
        drug_embedding = drug_batch.dt.view(batch, 3072).to(device)
        protein_embedding = protein_batch.pt.view(batch, 3072).to(device)
        tensors = [drug_graph, protein_graph, drug_embedding, protein_embedding]
        x = self.fusion(tensors)
        x = x.mean(dim=1)
        re = x
        for j in range(self.layer_output-1):
            x = torch.tanh(self.W_out[j](x))
        predicted = self.W_interaction(x)
        return re, predicted

