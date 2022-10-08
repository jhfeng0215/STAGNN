
from torch import nn
from TCN.TCNcell import TemporalConvNet

import torch

class STformer(nn.Module):

    def __init__(self,n_graph, n_vertex):
        super().__init__()
        # self.tcn = TemporalConvNet(n_vertex, [n_vertex,200,n_vertex])
        # num_inputs, num_channels
        self.tcn_in = TemporalConvNet(n_vertex, [n_vertex,200,n_vertex])
        self.tcn_out = TemporalConvNet(n_vertex, [n_vertex, 200, n_vertex])
        self.elu = nn.ELU()
        self.linear = nn.Linear(n_graph,1)
        self.bn = nn.BatchNorm2d(n_vertex,affine=True)
        self.bn2 = nn.BatchNorm2d(n_vertex,affine=True)


    def forward(self,batch_size, n_graph, n_vertex,node_feature):

        node_feature = node_feature.permute(0,2,1,3)


        cheack_in = node_feature[:,:,:,0]
        cheack_in = self.tcn_in(cheack_in)
        cheack_in =  cheack_in.permute(0,2,1).unsqueeze(-1)


        cheack_out = node_feature[:,:,:,1]
        cheack_out = self.tcn_out(cheack_out)
        cheack_out = cheack_out.permute(0,2,1).unsqueeze(-1)


        x = torch.cat((cheack_in, cheack_out), 3)
        x = self.elu(x)


        return cheack_in
