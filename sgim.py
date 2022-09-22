import time
from turtle import forward
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.datasets import GEDDataset
from torch_geometric.data import DataLoader
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv,GCNConv,GINConv,Sequential
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.transforms import OneHotDegree
from torch_geometric.data import Batch
from torchinfo import summary
from scipy.stats import spearmanr, kendalltau
from torch_geometric.utils import softmax, degree, sort_edge_index,to_dense_batch
import torchvision.models as models
from torch_scatter import scatter
# from utils import init_weights
# from layers import *  
from torch_geometric.typing import OptTensor
from torch_geometric.nn.inits import ones, zeros
from typing import Optional
from torch.nn.modules.instancenorm import _InstanceNorm
import torch
from torch import Tensor
from torch_scatter import scatter_add,scatter_mean,scatter

from torch_geometric.utils import softmax
from torch.nn import Parameter,LayerNorm
import json

import os
import datetime
import argparse
import requests
import prettytable as pt
import inspect

class InstanceNorm(_InstanceNorm):
    r"""Applies instance normalization over each individual example in a batch
    of node features as described in the `"Instance Normalization: The Missing
    Ingredient for Fast Stylization" <https://arxiv.org/abs/1607.08022>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately for
    each object in a mini-batch.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, optional): The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`False`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses instance statistics in both training and eval modes.
            (default: :obj:`False`)
    """
    def __init__(self, in_channels, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False):
        super().__init__(in_channels, eps, momentum, affine,
                         track_running_stats)

    def forward(self, x: Tensor, batch: OptTensor = None) -> Tensor:
        """"""
        if batch is None:
            out = F.instance_norm(
                x.t().unsqueeze(0), self.running_mean, self.running_var,
                self.weight, self.bias, self.training
                or not self.track_running_stats, self.momentum, self.eps)
            return out.squeeze(0).t()

        batch_size = int(batch.max()) + 1

        mean = var = unbiased_var = x  # Dummies.

        if self.training or not self.track_running_stats:
            norm = degree(batch, batch_size, dtype=x.dtype).clamp_(min=1)
            norm = norm.view(-1, 1)
            unbiased_norm = (norm - 1).clamp_(min=1)

            mean = scatter(x, batch, dim=0, dim_size=batch_size,
                           reduce='add') / norm

            x = x - mean.index_select(0, batch)

            var = scatter(x * x, batch, dim=0, dim_size=batch_size,
                          reduce='add')
            unbiased_var = var / unbiased_norm
            var = var / norm

            momentum = self.momentum
            if self.running_mean is not None:
                self.running_mean = (
                    1 - momentum) * self.running_mean + momentum * mean.mean(0)
            if self.running_var is not None:
                self.running_var = (
                    1 - momentum
                ) * self.running_var + momentum * unbiased_var.mean(0)
        else:
            if self.running_mean is not None:
                mean = self.running_mean.view(1, -1).expand(batch_size, -1)
            if self.running_var is not None:
                var = self.running_var.view(1, -1).expand(batch_size, -1)

            x = x - mean.index_select(0, batch)

        out = x / (var + self.eps).sqrt().index_select(0, batch)

        if self.weight is not None and self.bias is not None:
            out = out * self.weight.view(1, -1) + self.bias.view(1, -1)

        return out


    def __repr__(self):
        return f'{self.__class__.__name__}({self.num_features})'

class GraphNorm(torch.nn.Module):
    r"""Applies graph normalization over individual graphs as described in the
    `"GraphNorm: A Principled Approach to Accelerating Graph Neural Network
    Training" <https://arxiv.org/abs/2009.03294>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} - \alpha \odot
        \textrm{E}[\mathbf{x}]}
        {\sqrt{\textrm{Var}[\mathbf{x} - \alpha \odot \textrm{E}[\mathbf{x}]]
        + \epsilon}} \odot \gamma + \beta

    where :math:`\alpha` denotes parameters that learn how much information
    to keep in the mean.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
    """
    def __init__(self, in_channels: int, eps: float = 1e-5):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(in_channels))
        self.mean_scale = torch.nn.Parameter(torch.Tensor(in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)
        ones(self.mean_scale)


    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """"""
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        batch_size = int(batch.max()) + 1

        mean = scatter_mean(x, batch, dim=0, dim_size=batch_size)
        out = x - mean.index_select(0, batch) * self.mean_scale
        var = scatter_mean(out.pow(2), batch, dim=0, dim_size=batch_size)
        std = (var + self.eps).sqrt().index_select(0, batch)
        return self.weight * out / std + self.bias


    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels})'


class LayerNorm(torch.nn.Module):
    r"""Applies layer normalization over each individual example in a batch
    of node features as described in the `"Layer Normalization"
    <https://arxiv.org/abs/1607.06450>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    The mean and standard-deviation are calculated across all nodes and all
    node channels separately for each object in a mini-batch.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
    """
    def __init__(self, in_channels, eps=1e-5, affine=True):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps

        if affine:
            self.weight = Parameter(torch.Tensor([in_channels]))
            self.bias = Parameter(torch.Tensor([in_channels]))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, batch: OptTensor = None) -> Tensor:
        """"""
        if batch is None:
            x = x - x.mean()
            out = x / (x.std(unbiased=False) + self.eps)

        else:
            batch_size = int(batch.max()) + 1

            norm = degree(batch, batch_size, dtype=x.dtype).clamp_(min=1)
            norm = norm.mul_(x.size(-1)).view(-1, 1)

            mean = scatter(x, batch, dim=0, dim_size=batch_size,
                           reduce='add').sum(dim=-1, keepdim=True) / norm

            x = x - mean.index_select(0, batch)

            var = scatter(x * x, batch, dim=0, dim_size=batch_size,
                          reduce='add').sum(dim=-1, keepdim=True)
            var = var / norm

            out = x / (var + self.eps).sqrt().index_select(0, batch)

        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias

        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels})'

def init_weights(net, init_type='kaiming', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # print(m)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)


class CNN(torch.nn.Module):
    def __init__(self, in_channels,hidden_size):
        super(CNN, self).__init__()
        # print(hidden_size)
        # hidden_size = 32
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=12, kernel_size=1, stride=1)
        self.ln1 = nn.InstanceNorm2d(12,affine=True)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=(1, 1))
        self.ln2 = nn.InstanceNorm2d(12,affine=True)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=(1, 1))
        self.ln3 = nn.InstanceNorm2d(12,affine=True)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=1, kernel_size=3, stride=1, padding=(1, 1))
        self.ln4 = nn.InstanceNorm2d(1,affine=True)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # x = 
        # print(x.shape)
        x = self.ln1(F.leaky_relu(self.conv1(x)))
        x = self.ln2(F.leaky_relu(x + self.conv2(x)))
        x = self.ln3(F.leaky_relu(x + self.conv3(x)))
        x = self.ln4(F.relu(self.conv4(x)))
        return self.flatten(x)
                # print(x.shape)
        # x = F.leaky_relu(self.conv1(x))
        # x = F.leaky_relu(x + self.conv2(x))
        # x = F.leaky_relu(x + self.conv3(x))
        # x = F.relu(self.conv4(x))
        # return self.flatten(x)




class CNN_reg(nn.Module):
    def __init__(self,in_channels):
        super(CNN_reg,self).__init__()
        self.cnn_head = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 16,  kernel_size=32, stride=1,groups=2),
            # nn.InstanceNorm2d(16),
            # nn.BatchNorm2d(16),
            # nn.MaxPool2d(2,2),
            # nn.LayerNorm([16,32,32],elementwise_affine=False),
            nn.ReLU(inplace = True)
        )
        self.cnn_last = nn.Sequential(
                    nn.Conv2d(in_channels = 16, out_channels = 64,kernel_size=7, stride=1,groups=16),
                    # nn.InstanceNorm2d(16),
                    # nn.LayerNorm([64,32,32],elementwise_affine=False),
                    # nn.BatchNorm2d(64),
                    # nn.MaxPool2d(2,2),
                    nn.ReLU(inplace = True),
                    # nn.AdaptiveAvgPool2d((2,2)),
                    nn.Flatten()
        )
        # self.cnn = nn.ModuleList([])
        # for i in range(3):
        #     self.cnn.append(        
        #         nn.Sequential(
        #             nn.Conv2d(in_channels = 16, out_channels = 16,kernel_size=3, stride=1, padding=(1, 1),groups=16),
        #             nn.InstanceNorm2d(16),
        #             nn.LeakyReLU(0.2)
        #             )
        #         )
    def forward(self,x):
        x = self.cnn_head(x)
        # for cnn_layers in self.cnn:
        #     x = x + cnn_layers(x)
            # x = cnn_layers(x)
        x = self.cnn_last(x)

        return x

def add_pool(x,batch):
    x = to_dense_batch(x,batch)[0]
    x = x.sum(-2)
    return x

# def activate_function(x):
#     if x < 0

class GNN_layers(nn.Module):
    def __init__(self,in_channels,gnn_type):
        super(GNN_layers,self).__init__()
        

        if gnn_type == 'gin':
            self.nns = nn.Linear(in_channels,in_channels)
            self.gnn_layers = Sequential('x,edge_index,batch',[
                (GINConv(self.nns,eps = True),'x,edge_index -> x'),
                (LayerNorm(in_channels),'x,batch -> x'),
                nn.ReLU(inplace=True)
            ])
        elif gnn_type == 'sage':
            self.gnn_layers = Sequential('x,edge_index,batch',[
                (SAGEConv(in_channels,in_channels),'x,edge_index -> x'),
                (LayerNorm(in_channels),'x,batch -> x'),
                nn.ReLU(inplace=True)
            ])
        elif gnn_type == 'gcn':
            self.gnn_layers = Sequential('x,edge_index,batch',[
                (GCNConv(in_channels,in_channels),'x,edge_index -> x'),
                (LayerNorm(in_channels),'x,batch -> x'),
                nn.ReLU(inplace=True)
            ])

        self.ffn = Sequential('x,batch',[
            (nn.Linear(in_channels,in_channels),'x -> x'),
            (LayerNorm(in_channels),'x,batch -> x'),
            nn.ReLU(inplace=True)
            ]
        )

    def forward(self,x,edge_index,batch):
        # x =  self.gnn_layers(x,edge_index,batch) 
        x = x + self.gnn_layers(x,edge_index,batch)
        # x = x + self.ffn(x,batch)
        x = self.ffn(x,batch)
        # x = self.norm(self.ffn(x,batch) + x,batch)
        # print(x.size())
        graph_emds = global_add_pool(x,batch).clone()
        # graph_emds = self.gb_readout(x,batch).clone()
        node_emds = to_dense_batch(x,batch)[0].clone()
        node_emds = torch.matmul(node_emds.transpose(1,2),node_emds)
        # print(node_emds.size())
        return x,graph_emds,node_emds

class Backbone(nn.Module):
    def __init__(self,in_channels,gnn_num,gnn_type):
        super(Backbone,self).__init__()
        self.gnn_num = gnn_num
        self.gnns = nn.ModuleList([GNN_layers(in_channels,gnn_type) for i in range(self.gnn_num)])

    def forward(self,x,edg_index,batch):
        node_emds = to_dense_batch(x,batch)[0]
        node_emds = torch.matmul(node_emds.transpose(1,2),node_emds).clone()
        gb,nb = [global_add_pool(x,batch).clone()],[node_emds]
        for layers in self.gnns:
            x,graph_emds,node_emds = layers(x,edg_index,batch)
            # print(len(to_dense_batch(x,batch)))
            gb.append(graph_emds)
            nb.append(node_emds)
        # print(len(nb),nb[0].size())
        nb = torch.stack(nb,dim = 1)  # [batch,gnn_num+1,hidden_channels,hidden_channels]
        gb = torch.cat(gb,dim = -1)    # [batch,(gnn_num+1)*hidden_channels]

        return gb,nb

class SGim(nn.Module): # 3.8VERSION
    def __init__(self,in_channels,args):

        super(SGim, self).__init__()

        self.args = args
        self.in_channels = in_channels
        self.gnn_num = self.args.num_gnn
        self.out_channels = 1
        self.hidden_channels = self.args.hidden_size
        self.gnn_type = self.args.gnn_type
        # self.mode = 'self'


        self.weight = torch.nn.Parameter(torch.FloatTensor(self.hidden_channels,self.hidden_channels))
        self.softmax = nn.Softmax2d()
        self.embedding = nn.Linear(in_channels,self.hidden_channels)

        self.backbone = Backbone(self.hidden_channels,self.gnn_num,self.gnn_type)

        # if self.mode == 'self':
        #     in_channels = self.gnn_num + 1
        # elif self.mode == 'res':
        #     in_channels = self.gnn_num - 1
        #     # in_channels = 1
        # elif self.mode == 'dense':
        #     in_channels = 0
        #     for idx_i in range(self.gnn_num + 1):
        #         for idx_j in range(idx_i,self.gnn_num + 1):
        #             in_channels = in_channels + 1

        # self.cnns = nn.ModuleList([CNN_reg(in_channels) for i in range(3)])
        self.cnns = CNN(self.gnn_num + 1,self.hidden_channels)

        if self.args.mode == 'graph':
            reg_channels = (self.gnn_num + 1)*self.hidden_channels *2
        elif self.args.mode == 'node':
            reg_channels = self.hidden_channels*self.hidden_channels
        elif self.args.mode == 'full':
            reg_channels = (self.gnn_num + 1)*self.hidden_channels *2 + self.hidden_channels*self.hidden_channels


        self.reg = nn.Sequential(            
            # nn.Linear((self.gnn_num + 1)*self.hidden_channels *2,self.hidden_channels // 2 ),
            # nn.Linear(self.hidden_channels*self.hidden_channels,self.hidden_channels // 2 ),
            # nn.Linear((self.gnn_num + 1)*self.hidden_channels *2 +self.hidden_channels*self.hidden_channels ,self.hidden_channels // 2 ),
            nn.Linear(reg_channels,self.hidden_channels // 2),
            nn.LayerNorm(self.hidden_channels // 2),
            nn.ReLU(inplace = True),
            # nn.Linear(self.hidden_channels //2,self.hidden_channels // 4 ),
            # nn.LayerNorm(self.hidden_channels // 4),
            # nn.ReLU(inplace = True),
            nn.Linear(self.hidden_channels // 2,self.out_channels),
            # nn.Sigmoid()
            # nn.LeakyReLU(0.1,inplace = True)
            # nn.ReLU(inplace=True)
            )

        self.reset_parameters()
        self.apply(init_weights)
    def forward(self,data1,data2):
        x_t = data1.x
        x_s = data2.x
        edge_index_t = data1.edge_index
        edge_index_s = data2.edge_index
        batch_t = data1.batch
        batch_s = data2.batch
        x_i = self.embedding(x_t)
        x_j = self.embedding(x_s)
        edge_index_i = edge_index_t
        edge_index_j = edge_index_s
        batch_i = batch_t
        batch_j = batch_s

        gb_i,nb_i = self.backbone(x_i,edge_index_i,batch_i)
        gb_j,nb_j = self.backbone(x_j,edge_index_j,batch_j)
        
        # 128*2
        gb = self.gb_ops(gb_i,gb_j) 
        # 256 1024
        nb = self.nb_ops(nb_i,nb_j) if self.args.drop_att else self.nb_ops_without_attention(nb_i,nb_j)
        # print(x)
        # print(nb.shape)
        # print(nb)
         
        if self.args.mode == 'node':
            nb = self.cnns(nb)
            x = nb
        elif self.args.mode == 'graph':
            x = gb
        elif self.args.mode == 'full':
            nb = self.cnns(nb)
            x = torch.cat( [gb,nb ],dim = 1)
        # print(nb)
        # x = nb
        # print(gb.shape)
        # print(nb.shape)
        # print(x.shape)
        x = self.reg(x).squeeze(-1)
        # print(x)
        return x

    def gb_ops(self,x_s,x_t):
        # return torch.abs(x_s - x_t)
        return torch.cat([x_s,x_t],dim = 1)

    def nb_ops(self, x_s, x_t):

        x = torch.matmul(x_s,x_t.transpose(2,3))
        att = torch.abs(x_s - x_t)
        att = torch.matmul(att,self.weight.t())
        att = self.softmax(att)
        return F.relu(att*x)

    def nb_ops_without_attention(self, x_s, x_t):

        x = torch.matmul(x_s,x_t.transpose(2,3))
        return F.relu(x)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self):
        return 'SGim'
    # def __repr__(self):
    #     return 'SFGNNGraphSim'

