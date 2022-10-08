import torch
from torch import nn
from Layers.GATmodels import GAT
from Layers.gnnlayer import SharedGraphLayer
from TCN.tcn_graph import STformer

class STAblock(nn.Module):

    def __init__(self, batch_size,n_graph,gat_hidden, n_vertex,gat_heads,
                 device,inputsize,alpha,dropout,gnn_kernel,gnn_hidden):
        super(STAblock,self).__init__()

        self.GAT_in = GAT(batch_size,n_graph,n_vertex,inputsize, gat_hidden, inputsize, dropout,alpha, gat_heads)
        self.GAT_out = GAT(batch_size, n_graph,n_vertex,inputsize, gat_hidden, inputsize, dropout, alpha, gat_heads)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU()
        self.tcn = STformer(n_graph, n_vertex)
        self.batch_size=batch_size
        self.n_graph = n_graph
        self.n_vertex = n_vertex
        self.sharedgnn = SharedGraphLayer(batch_size,n_graph,n_vertex, 5, gnn_hidden, device, alpha, gnn_kernel)
        self.device = device
        self.conv1 = nn.Conv1d(n_vertex, n_vertex,5 , padding=2)

    def forward(self,all_node_feature,all_node_edges):

        # 1. GAT
        gat_output_in = self.GAT_in(all_node_feature,all_node_edges)
        gat_output_out = self.GAT_out(all_node_feature, all_node_edges.permute(0,1,3,2))
        gat_output = torch.cat([gat_output_in,gat_output_out],dim=-1)


        # 2.TCN
        node_new_feature = self.tcn(self.batch_size, self.n_graph,  self.n_vertex,all_node_feature)


        # # 3.concat

        node_new_feature = torch.cat([node_new_feature, all_node_feature,gat_output], dim=-1)
        temp = node_new_feature

        #
        # # 4.Sharedgnn
        sharedgnn_output = self.sharedgnn(node_new_feature,all_node_edges)


        # 5.Sharedmlp
        node_new_feature = node_new_feature.permute(0,2,1,3).reshape(self.batch_size,self.n_vertex,-1)
        sharedmlp_output = self.conv1(node_new_feature)
        sharedmlp_output = sharedmlp_output.reshape(self.batch_size,self.n_vertex,self.n_graph,-1).permute(0, 2, 1, 3)


        # 6.Concat
        output = torch.cat([sharedgnn_output,sharedmlp_output,temp],dim=-1)


        return output
