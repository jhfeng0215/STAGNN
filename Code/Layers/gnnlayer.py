import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedGraphLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    # self.GAT = GAT(batch_size, n_graph, 2, 10, 2, 0.2, 0.01, 2)
    def __init__(self, batch_size,n_graph,n_vertex, in_features, out_features, device, alpha, num_kernel):
        super(SharedGraphLayer, self).__init__()


        self.W = nn.Parameter(torch.empty(size=(n_graph,in_features, out_features)).to(device),requires_grad=True)
        self.O = nn.Parameter(torch.empty(size=(n_graph,out_features, in_features)).to(device),requires_grad=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(num_kernel,2)
        self.batch_size = batch_size
        self.n_graph = n_graph
        self.n_vertex = n_vertex

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.conv1 = nn.Conv2d(n_graph, n_graph, (5, 5), padding=(2, 0))
        self.conv2 = nn.Conv2d(n_graph, n_graph, (5, 5), padding=(2, 0))


    def forward(self, h, adj):

        # adj_trans
        adj = torch.sum(adj,dim=1)
        adj = torch.where(adj>=1,1,0)

        # feature liner trasfer
        Wh = torch.einsum('ijkh,khg->ijkg', [h.permute(0,2,1,3), self.W]).permute(0,2,1,3)
        # Wh = torch.matmul(h, )  # h.shape: (batch_size, n_graph, N, in_features), Wh.shape: (batch_size, n_graph,N, out_features)

        Wh = Wh.permute(0,2,1,3).reshape(self.batch_size,self.n_vertex,-1)
        Wh = self.leakyrelu(Wh)
        node_feature = torch.einsum('ijh,ihg->ijhg', [adj, Wh])

        feature_in = torch.mean(node_feature,dim=1)
        feature_out = torch.mean(node_feature, dim=2)
        # node_feature_in = Wh * adj
        feature_in = self.leakyrelu(feature_in)
        feature_in = feature_in.reshape(self.batch_size,self.n_vertex,self.n_graph,-1).permute(0,2,1,3)


        # 对不同Graph进行卷积
        feature_in = self.conv1(feature_in)
        # node_feature = self.leakyrelu(node_feature)

        feature_out = self.leakyrelu(feature_out)
        feature_out = feature_out.reshape(self.batch_size, self.n_vertex, self.n_graph, -1).permute(0, 2, 1, 3)

        # 对不同Graph进行卷积
        feature_out = self.conv2(feature_out)

        node_feature = torch.cat([feature_in,feature_out],dim=-1)



        return node_feature






