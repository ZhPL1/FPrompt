# FPrompt

This is the official implement of FPrompt. 

## Environments

python==3.9

pytorch==1.13

pytorch-cuda==11.7

pyg==2.5.3


## Datasets

We include benchmark dataset pokec_n, pokec_z, and credit in the **data** folder. Furthermore, we release a new benchmark. The benchmark consists of four datasets, all created by sampling from the Pokec social network data based on geographic regions [1].

## Pre-Trained Model

We saved a pre-trained GNN model named pokec_n.pt, which is obtained via Infomax. The GNN architecture can be found in **models.py**. You can pre-train your own GNN model via additional pre-training methods with the same architecture. For the implement of Infomax and other methods, we refer readers to https://pyg.org.

## Run FPrompt

python -m main

[1] Lubos Takac and Michal Zabovsky. 2012. Data analysis in public social networks. *In International scientific conference and international workshop present day trends of innovations*, Vol. 1.
