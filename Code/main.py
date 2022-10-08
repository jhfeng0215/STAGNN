import torch.functional as F
import torch.optim as optim
from conf import *
from models import STA
import load_data
import torch
import torch.nn as nn
import random
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from tqdm import tqdm, trange
import warnings
from metrics import calculate_metrics
warnings.filterwarnings('ignore')
import time



#
# viz = Visdom()
# viz.line([[0.,0.,0.,0.]], [0], win='train', opts=dict(title='loss&acc', legend=['train_loss', 'train_rmse', 'train_mae', 'train_mape']))
# viz.line([[0.,0.,0.,0.]], [0], win='val', opts=dict(title='loss&acc', legend=['val_loss', 'val_rmse', 'val_mae', 'val_mape']))


model = STA(batch_size,n_graph,gat_hidden, n_vertex,gat_heads,
                 device,inputsize,alpha,dropout,gnn_kernel,gnn_hidden,n_block).to(device)


# count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')



def initialize_weights(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.kaiming_uniform(p.data,1.414)
model.apply(initialize_weights)


random.seed(seed)
print('dataset initializing start')
train_iterator, val_iterator,tst_iterator = load_data.data_process(path,batch_size,device,day,hour,week)

optimizer = ''
# optimizer
if opt == "rmsprop":
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
elif opt == "adam":
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=adam_eps)
elif opt == "adamw":
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=adam_eps)



scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 mode= 'min',
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience,
                                                 threshold=1e-3)
huberloss = nn.HuberLoss(reduction='mean',delta= 0.75)
mseloss = nn.MSELoss(reduction='mean')
sml1loss = nn.SmoothL1Loss(reduction='mean')
l1loss = nn.L1Loss(reduction='mean')



def train(epoch,la):

    epoch_loss = 0
    pre = []
    real = []
    for train_node_feature in train_iterator:

        train_node_feature= train_node_feature
        train_lab = [data[-1][0].x for data in train_node_feature]

        optimizer.zero_grad()
        train_lab = torch.stack(train_lab)
        pre_lab = model(batch_size, n_graph,n_vertex, train_node_feature)
        loss = la * l1loss(pre_lab, train_lab) + (1-la) * mseloss(pre_lab, train_lab)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        train_lab = train_lab.reshape(batch_size,-1).data.cpu()
        pre_lab = pre_lab.reshape(batch_size,-1).data.cpu()
        pre.append(pre_lab)
        real.append(train_lab)

    pre = torch.stack(pre).numpy()
    real = torch.stack(real).numpy()

    mse = mean_squared_error(real.reshape(1,-1), pre.reshape(1,-1))
    mae = mean_absolute_error(real.reshape(1,-1), pre.reshape(1,-1))
    rmse = np.sqrt(mean_squared_error(real.reshape(1,-1), pre.reshape(1,-1)))



    return epoch_loss/len(train_iterator) , mse,  mae, rmse



def val(la):


    epoch_loss = 0
    pre = []
    real =[]
    for  val_node_feature in val_iterator:

        val_lab = [data[-1][0].x for data in val_node_feature]

        val_lab = torch.stack(val_lab)
        pre_lab = model(batch_size, n_graph,n_vertex, val_node_feature)
        loss = la * l1loss(pre_lab, val_lab) + (1 - la) * mseloss(pre_lab, val_lab)
        epoch_loss += loss.item()


        val_lab = val_lab.reshape(batch_size,-1).data.cpu()
        pre_lab = pre_lab.reshape(batch_size,-1).data.cpu()
        pre.append(pre_lab)
        real.append(val_lab)

    pre = torch.stack(pre).numpy()
    real = torch.stack(real).numpy()

    mse = mean_squared_error(real.reshape(1, -1), pre.reshape(1, -1))
    mae = mean_absolute_error(real.reshape(1, -1), pre.reshape(1, -1))
    rmse = np.sqrt(mean_squared_error(real.reshape(1, -1), pre.reshape(1, -1)))

    return epoch_loss/len(val_iterator) , mse,  mae, rmse




def tst(la):

    val_MSE = []
    val_RMSE = []
    val_MAE = []
    epoch_loss = 0
    pre = []
    real =[]
    for  tst_node_feature in tst_iterator:

        tst_lab = [data[-1][0].x for data in tst_node_feature]

        tst_lab = torch.stack(tst_lab)
        pre_lab = model(batch_size, n_graph,n_vertex, tst_node_feature)
        loss = la * l1loss(pre_lab, tst_lab) + (1 - la) * mseloss(pre_lab, tst_lab)
        epoch_loss += loss.item()


        tst_lab = tst_lab.reshape(batch_size,-1).data.cpu()
        pre_lab = pre_lab.reshape(batch_size,-1).data.cpu()
        pre.append(pre_lab)
        real.append(tst_lab)

    pre = torch.stack(pre).numpy()
    real = torch.stack(real).numpy()

    mse = mean_squared_error(real.reshape(1, -1), pre.reshape(1, -1))
    mae = mean_absolute_error(real.reshape(1, -1), pre.reshape(1, -1))
    rmse = np.sqrt(mean_squared_error(real.reshape(1, -1), pre.reshape(1, -1)))

    return epoch_loss/len(tst_iterator) , mse,  mae, rmse



def run(total_epoch,la):

    for epoch in tqdm(range(total_epoch),desc='Epoch'):
        loss ,  mse,  mae, rmse= train(epoch,la)
        print('\n')
        print('[Epoch: %3d/%3d] Tra Loss: %.7f, Tra MSE: %.4f, Tra MAE: %.4f, Tra RMSE: %.4f'
            % (epoch+1 , total_epoch, loss , mse,  mae, rmse))



        loss ,  mse,  mae, rmse = val(la)
        print(
            '[Epoch: %3d/%3d] Val Loss: %.7f, Val MSE: %.4f, Val MAE: %.4f, Val RMSE: %.4f'
            % (epoch+1 , total_epoch, loss , mse, mae, rmse))
        scheduler.step(loss)


        loss ,  mse,  mae, rmse = tst(la)
        print(
            '[Epoch: %3d/%3d] Test Loss: %.7f, Test MSE: %.4f, Test MAE: %.4f, Test RMSE: %.4f'
            % (epoch+1 , total_epoch, loss , mse, mae, rmse))
        scheduler.step(loss)


if __name__ == '__main__':
    run(total_epoch=epochs,la =loss_alpha)
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)




