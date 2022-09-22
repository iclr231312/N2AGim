import time
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.datasets import GEDDataset
from torch_geometric.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv,GCNConv,GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.transforms import OneHotDegree
from torch_geometric.data import Batch
from torchinfo import summary
from scipy.stats import spearmanr, kendalltau
from torch_geometric.utils import softmax, degree, sort_edge_index,to_dense_batch
import torchvision.models as models

from utils import *

import os
import datetime
import argparse
import requests
import prettytable as pt
import inspect


class GNNModel(nn.Module):
    """
    GNN model,
    input : x, edge_index, batch
    output: list of [batch, node, channels]
    """
    def __init__(self, in_channels, out_channels, gnn_num, return_mode = 'dense'):
        super(GNNModel, self).__init__()

        assert return_mode in ['dense','sparse']
        self.return_mode = return_mode

        self.gnn = []
        self.first_head = nn.Linear(in_channels,out_channels)
        for i in range(gnn_num):
            self.nns = nn.Sequential(nn.Linear(out_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(),
                            nn.Linear(out_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU())
            self.gnn.append(GINConv(self.nns))
        self.gnn = torch.nn.ModuleList(self.gnn)

    def forward(self,x, edge_index, batch):

        #first embedding the input graph
        x = self.first_head(x)

        if self.return_mode == 'dense':
            var_temp = [to_dense_batch(x, batch)[0]]

            #apply to gnn
            for idx in range(len(self.gnn)):

                x = self.gnn[idx](x,edge_index)
                var_temp.append(to_dense_batch(x, batch)[0])

        elif self.return_mode == 'sparse':
            var_temp = [x]

            #apply to gnn
            for idx in range(len(self.gnn)):
                x = self.gnn[idx](x,edge_index)
                var_temp.append(x)


        return var_temp


class ReadOutBaseModel(nn.Module):
    """
    a simple readout baseline, which is defined as:
        graph_representation = rho( \Box_{i \in v} \phi( x_i ) )
    where \Box can be donated as max pooling, mean pooling and sum pooling.

    Args:
        in_channels:
            the channels of the input, the input should be of shape [batch, in_channels]
        out_channels
            the channels of the output
        mode:
            ['max','mean','sum'], indices which pooling methods to used;
        
    Return:
        a graph representation of shape [batch, out_channels]
    """

    def __init__(self, in_channels, out_channels, mode = 'sum'):
        super(ReadOutBaseModel,self).__init__()
        assert mode in ['max','mean','sum']
        self.mode = mode
        self.phi = nn.Sequential(
                nn.Linear(in_channels,out_channels // 2),
                nn.BatchNorm1d(out_channels // 2),
                nn.ReLU()
        ) 
        
        self.rho = nn.Sequential(
                nn.Linear(out_channels // 2 , out_channels),
                nn.BatchNorm1d(out_channels),
                nn.Sigmoid()
        )

    def forward(self,x,edge_index,batch):

        #[num_nodes, inchannels] -> [num_nodes, outchannels]
        x = self.phi(x)

        if self.mode == 'sum':
            x = global_add_pool(x,batch)
        elif self.mode == 'max':
            x = global_max_pool(x,batch)
        elif self.mode == 'mean':
            x = global_mean_pool(x,batch)

        
        return self.rho(x)

class ReadOutMLPBaseline(nn.Module):
    """
    class of the ReadOut Baseline, define how to perform readout on the input of [num,[batch, num_nodes, channels]]
    Args:
            x:
                the shape of the x should be [num,[batch, num_nodes, channels]],
                and x is a list of Tensor [batch, num_nodes, channels]
    Returns:
            Tensor of shape [batch,out_channels]
    """
    def __init__(self, num, in_channels, out_channels, mode = 'sum'):
        super(ReadOutMLPBaseline,self).__init__()
        self.readout = []
        for i in range(num):
            self.readout.append(ReadOutBaseModel(in_channels, out_channels, mode))
        self.readout = nn.ModuleList(self.readout)
    
    def forward(self,x,edge_index,batch):

        result = []
        for idx,temp_x in enumerate(x):
            result.append(self.readout[idx](temp_x,edge_index,batch))
        result = torch.stack(result,dim = 1)
        return torch.sum(result,dim = 1)
        



class SecondOrderPooling(nn.Module):
    """
    a class to perform second order pooling on the node embedding set to representaion a graph,
    which is defined as:
            graph_representation = X^{T} X

    Args:
        mode:
            ['self','res','dense'] indice the mode geraneated the X^{T} X
        x:
            the shape of the x should be [num,[batch, num_nodes, channels]],
            and x is a list of Tensor [batch, num_nodes, channels]

    Return:
        Tensor of the graph representation, which is of the shape [batch, num, channels, channels]
    """
    def __init__(self, mode):
        
        assert mode in ['self','res','dense']
        super(SecondOrderPooling,self).__init__()
        self.mode = mode

    def forward(self, x, edge_index, batch):
        # temp = []
        # for x_temp in x:
        #     print(x_temp.shape)
        #     temp.append(to_dense_batch(x_temp,batch)[0])

        if self.mode == 'self':
            temp = torch.stack(x,dim = 1)
            representaion = torch.matmul(temp.transpose(2,3),temp)

        elif self.mode == 'res':

            temp_i = x.copy()
            temp_j = x.copy()

            temp_i.pop(-1)
            temp_j.pop(0)

            temp_i = torch.stack(temp_i,dim = 1)
            temp_j = torch.stack(temp_j,dim = 1)

            representaion = torch.matmul(temp_i.transpose(2,3),temp_j)

        elif self.mode == 'dense':
            representaion = []
            for idx_i in range(len(x)):
                for idx_j in range(idx_i,len(x)):
                    representaion.append(torch.matmul(x[idx_i].transpose(1,2),x[idx_j]))
            representaion = torch.stack(representaion,dim = 1)

        return representaion
