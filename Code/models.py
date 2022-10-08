from STAblock import STAblock
import torch.nn as nn
import torch


class STA(nn.Module):
    def __init__(self, batch_size,n_graph,gat_hidden, n_vertex,gat_heads,
                 device,inputsize,alpha,dropout,gnn_kernel,gnn_hidden,n_layers):
        """Dense version of GAT."""
        super(STA, self).__init__()

        self.device = device
        self.STAblocks = nn.ModuleList([STAblock(batch_size,n_graph,gat_hidden, n_vertex,gat_heads,
                 device,inputsize,alpha,dropout,gnn_kernel,gnn_hidden) for _ in range(n_layers)])
        self.relu = nn.ReLU()
        self.linear_o = nn.Linear(n_graph,1)
        self.linear_1 = nn.Linear(12*n_layers, 2)


    def forward(self,batch_size, n_graph, n_vertex,input_graphs):




        # 0. Process the graph
        batch_graphs = [x[:] for x in input_graphs]
        features = []
        edge = []


        all_node_edges = []
        # batch
        all_node_feature = []
        weather_feature = []
        for i in range(len(batch_graphs)):
            demo_graph_edges= []
            demo_graph_features = []
            demo_weather_features = []
            for j in range(len(batch_graphs[i])):

                data = batch_graphs[i][j][0]
                adj_mtx = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.shape[1]).to(self.device), (n_vertex, n_vertex)).to_dense()
                demo_graph_edges.append(adj_mtx)
                demo_graph_features.append(data.x)
            all_node_feature.append(torch.stack(demo_graph_features))
            all_node_edges.append(torch.stack(demo_graph_edges))
        all_node_edges = torch.stack(all_node_edges)
        all_node_feature = torch.stack(all_node_feature)
        all_node_feature = all_node_feature[:,:-1,:,:]
        all_node_edges = all_node_edges[:,:-1,:,:]




        output = []
        for STAblock in self.STAblocks:
            output.append( STAblock(all_node_feature,all_node_edges))

        output = torch.stack(output,dim=-1).reshape(batch_size, n_graph, n_vertex,-1)


        output = self.linear_o(output.permute(0, 2, 3, 1)).squeeze()
        output = self.linear_1(output)
        return output