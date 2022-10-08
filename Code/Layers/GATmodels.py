import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers.layers import GraphAttentionLayer


class GAT(nn.Module):
    # self.GAT = GAT(batch_size, n_graph, 2, 10, 2, 0.2, 0.01, 2)
    def __init__(self, batch_size, n_graph,n_vertex,nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        # self.dropout = nn.Dropout2d()
        self.leakyrelu = nn.LeakyReLU(alpha)

        self.dropout = nn.Dropout2d(dropout)

        self.attentions = [GraphAttentionLayer(batch_size, n_graph,n_vertex,nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(batch_size, n_graph,n_vertex,nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=True)
        self.O = nn.Parameter(torch.empty(n_graph,nheads*nhid,1))

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout)

        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = self.dropout(x)
        x = torch.einsum('ijkh,khe->ijke', [x.permute(0,2,1,3), self.O]).permute(0,2,1,3)
        x = self.leakyrelu(x)
        return x

