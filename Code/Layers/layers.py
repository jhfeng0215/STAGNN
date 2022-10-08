import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    # self.GAT = GAT(batch_size, n_graph, 2, 10, 2, 0.2, 0.01, 2)
    def __init__(self, batch_size,n_graph,n_vertex, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)

        self.W = nn.Parameter(torch.zeros(size=(n_vertex, in_features, out_features)),requires_grad=True)
        self.a = nn.Parameter(torch.zeros(size=(n_vertex, 2 * out_features, 1)),requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.einsum('ijkh,khe->ijke', [h, self.W])# Wh.shape: (batch_size, n_graph,N, out_features)
        # Wh = torch.matmul(h, self.W)  # h.shape: (batch_size, n_graph, N, in_features), Wh.shape: (batch_size, n_graph,N, out_features)
        Wh = self.leakyrelu(Wh)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)   # zero_vec.shape: batch_size,N, N
        attention = torch.where(adj > 0, e, zero_vec)
        attention = self.softmax(attention)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return self.leakyrelu(h_prime)
        else:
            return self.leakyrelu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (batch_size, N, out_feature)
        # self.a.shape (batch_size,2 * out_feature, 1)
        # Wh1&2.shape (batch_size,N, 1)
        # e.shape (batch_size,N, N)
        Wh1 = torch.einsum('ijkh,khe->ijke', [Wh, self.a[:,:self.out_features, :]])
        Wh2 = torch.einsum('ijkh,khe->ijke', [Wh, self.a[:,self.out_features:, :]])

        # broadcast add
        e = Wh1 + Wh2.permute(0,1,3,2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

