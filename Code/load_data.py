import numpy as np
import torch

from conf import *
import random
from torch_geometric.data import Data
from torch_geometric.data import DataListLoader
# from torch.utils.data import DataLoader
import pandas as pd
from datetime import date,timedelta

# csv 格式
# ['start', 'end', 'node_feature', 'weather_feature', 'time']

def read_Data(df,pdt_time):
    day = 72
    hour = 3
    week = 72 * 7
    # pre = df[pdt_time:pdt_time + hour]
    # time1 = df[pdt_time - hour:pdt_time]
    # sample =[pre,time1]
    # for i in range(1,10):
    #     time3 = df[pdt_time - i*day :pdt_time - i*day + hour]
    #     time4 = df[pdt_time - i * day - hour:pdt_time - i * day ]
    #     sample.append(time3)
    #     sample.append(time4)

    pre = df[pdt_time:pdt_time + 1]
    time1 = df[pdt_time - 1:pdt_time]
    sample =[pre,time1]
    for i in range(1,8):
        time3 = df[pdt_time - i*day :pdt_time - i*day + 1]
        sample.append(time3)

    sample.reverse()
    data_list= []

    for period in sample:
        for index, row  in period.iterrows():
            data = Data(x = torch.tensor(row['node_feature'],dtype=torch.float32,requires_grad=True).to(device),
                        edge_index= torch.tensor(row['edge'],dtype=torch.int64).to(device))
            data_list.append([data])

    return data_list







def load_Data(train_node_feature,val_node_feature,tst_node_feature,device):


    train_iterator = DataListLoader(
        dataset=train_node_feature,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )


    val_iterator = DataListLoader(
        dataset=val_node_feature,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    tst_iterator = DataListLoader(
        dataset=tst_node_feature,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    return  train_iterator, val_iterator,tst_iterator


def split_Data(df,batch_size,device,day,hour,week):

    # 切分数据集



    df_train = df[0: int(len(df)*0.7)]
    df_validate = df[0:int(len(df)*0.9)]
    df_test = df[0:len(df)]

    day = day
    hour = hour
    week = week

    # train_set
    first_time = 2*week + 1
    train_adjacent =[]
    train_node_feature=[]
    train_weather_feature=[]
    train_time = True
    i = 0
    while train_time:
        time = first_time + i*1
        i = i + 1
        if time < len(df_train)-hour:
            node_feature =read_Data(df_train,time)
            train_node_feature.append(node_feature)

        else:
            train_time =False

    # Validate_Set
    first_time = len(df_train)
    val_adjacent =[]
    val_node_feature=[]
    val_weather_feature=[]
    val_time = True
    i = 0
    while val_time:
        time = first_time + i*1
        i = i + 1
        if time < len(df_validate)-hour:
            node_feature =read_Data(df_validate,time)
            val_node_feature.append(node_feature)
        else:
            val_time =False



    # Test_Set
    first_time = len(df_validate)
    tst_adjacent =[]
    tst_node_feature=[]
    tst_weather_feature=[]
    tst_time = True
    i = 0
    while tst_time:
        time = first_time + i*1
        i = i + 1
        if time < len(df_test)-hour:
            node_feature =read_Data(df_test,time)
            tst_node_feature.append(node_feature)
        else:
            tst_time =False

    train_iterator, val_iterator,test_iterator = load_Data(train_node_feature,val_node_feature,tst_node_feature,device)

    return train_iterator, val_iterator, test_iterator





def normal(x):
    mean = x.mean(-1,keepdims=True)
    std = x.std(-1, keepdims=True)
    return (x - mean)/std


def data_process(path,batch_size,device,day,hour,week):
    df = pd.read_csv(path)
    df['node_feature'] = df['node_feature'].apply(lambda x: eval(x))
    df['edge'] = df[['start','end']].apply(lambda x: [eval(x[0]),eval(x[1])],axis= 1)


    train_iterator, val_iterator,test_iterator = split_Data(df,batch_size,device,day,hour,week)

    return train_iterator, val_iterator,test_iterator