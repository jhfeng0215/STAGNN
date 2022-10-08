import torch

# globel train param
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# data_path
path = r'D:\研究生项目\共享单车\模型数据集\原始数据\预处理数据\Divvy_20min_2021_rep.csv'

day = 72
hour = 3
week = 72 * 7

loss_alpha = 0.35

epochs = 200
n_graph = 8
n_vertex = 440
n_block = 1
dropout =0.2
batch_size = 16
lr=0.001
seed = 42


inputsize = 2
alpha = 0.01


# gnn
gnn_kernel = 2
gnn_hidden = 5


# gat
gat_heads = 1
gat_hidden = 5
gat_layers =1

# ReduceLROnPlateau
factor= 0.1
patience=3

# optimizer ["rmsprop",'adam','adamw','adabound']

opt = 'adam'
adam_eps = 1e-3
weight_decay = 5e-4
delta= 1
