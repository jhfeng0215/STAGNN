# STAGNN: Spatial-Temporal Aggregated Graph Neural Network for Docked Shared-Bike Prediction

## Introduce
Spatial-Temporal Aggregated Graph Neural Network (STAGNN), combined with the global information extraction layer and local information extraction layer, to capture the global and local information from dynamic and directed shared bike network. 


## Frame of STAGNN

![dataset](/Figure/model.jpg)


## Dataset
We verify the performance of STAGNN on four real-world datasets: the Citi dataset, the Divvy dataset, the Captial dataset, and the Bay dataset.


![dataset](/Figure/dataset.jpg)


Due to datasets is very large, you could download the dataset at:https://pan.baidu.com/s/102uJXjrH2ERLZ4ojcsAboA . Verification Code：18mt

## Train and Test
We conduct experiments on the four real-world datasets. Each dataset is divided into training, validation, and test sets according to 7:2:1. We generate the time sequence by sliding a window of width 7 + 1, where 7 represents the number of historical snapshots at the same time, and 1 represents the predicted number of times. In STAGNN, we set the graph hidden dimensions to 10, the graph attention head to 2, and the negative slope of all Leakyrelu functions to 0.01. We use three hidden temporal convolution layers, and the hidden size is the number of stations. The Adam optimizer \cite{kingma2014adam} is utilized to train the model. The maximum training iteration is set to 200.



## Result
The experiment results for predicting the number of shared bikes as follows. We can see that STAGNN outperforms all baseline methods in two evaluation metrics on four datasets with different time slots.
![dataset](/Figure/result.jpg)
