# STAGNN: Spatial-Temporal Aggregated Graph Neural Network for Docked Shared-Bike Prediction

## Introduce
Spatial-Temporal Aggregated Graph Neural Network (STAGNN), combined with the global information extraction layer and local information extraction layer, to capture the global and local information from dynamic and directed shared bike network. 


## Frame of STAGNN
The frame of STAGNN is shown as follows. The key component of STAGNN is the spatial-temporal block,it could be divided into the global information extraction layer and the local information extraction layer.The global information extraction layer includes the dynamic graph attention network and Temporal Convolutional Network. The former could identify the dynamic spatial correlation among stations, and the latter is utilized to capture the long temporal information. The shared GNN and shared MLP form the static information extraction layer. The shared GNN aims to capture the local spatial dependencies between stations, and the shared MLP tries to identify the near temporal information.


![dataset](/Figure/model.jpg)


## Dataset
We verify the performance of STAGNN on four real-world datasets: the Citi dataset, the Divvy dataset, the Captial dataset, and the Bay dataset.


![dataset](/Figure/dataset.jpg)


Due to datasets is very large, you could download the dataset at:https://pan.baidu.com/s/102uJXjrH2ERLZ4ojcsAboA . Verification Code：18mt

## Train and Test
We conduct experiments on the four real-world datasets. Each dataset is divided into training, validation, and test sets according to 7:2:1. We generate the time sequence by sliding a window of width 7 + 1, where 7 represents the number of historical snapshots at the same time, and 1 represents the predicted number of times. In STAGNN, we set the graph hidden dimensions to 10, the graph attention head to 2, and the negative slope of all Leakyrelu functions to 0.01. We use three hidden temporal convolution layers, and the hidden size is the number of stations. The Adam optimizer is utilized to train the model. The maximum training iteration is set to 200.


## Prerequisites
Our code is based on Python3.8. There are a few dependencies to run the code. The major libraries are listed as follows:
- torch-geometric  2.0.4
- torch 1.11.0
- scikit-learn 1.0.2



## Result
The experiment results for predicting the number of shared bikes as follows. We can see that STAGNN outperforms all baseline methods in two evaluation metrics on four datasets with different time slots.
![dataset](/Figure/result.jpg)

## Updates

Oct. 8, 2022
- The STAGNN code
- The STAGNN data
