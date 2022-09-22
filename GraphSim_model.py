from tqdm import tqdm
import torch
from torch_geometric.datasets import GEDDataset
from torch_geometric.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv,GCNConv
from torch_geometric.transforms import OneHotDegree
from torch_geometric.data import Batch
from torchinfo import summary
from scipy.stats import spearmanr, kendalltau
from torch_geometric.utils import softmax, degree, sort_edge_index,to_dense_batch
import os
import argparse
import requests
import prettytable as pt


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels,in_channels,GNNModel,reshape=30):
        super(GNN, self).__init__()
        self.conv1 = GNNModel(in_channels, hidden_channels)
        self.conv2 = GNNModel(hidden_channels, hidden_channels)
        self.conv3 = GNNModel(hidden_channels, hidden_channels)
        self.reshape = reshape

    def Maxpadding_and_Resizing(self, x_t, batch_t, x_s, batch_s, max_num_nodes):
        x_t = to_dense_batch(x_t, batch=batch_t, max_num_nodes=max_num_nodes)[0]
        x_s = to_dense_batch(x_s, batch=batch_s, max_num_nodes=max_num_nodes)[0]

        x = torch.bmm(x_s, x_t.transpose(1, 2))
        x = nn.functional.interpolate(x.unsqueeze(1), size=self.reshape, mode='bilinear', align_corners=False)
        return x

    def node_embedding(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x1 = x.clone()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x2 = x.clone()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x3 = x.clone()
        return x1, x2, x3

    def forward(self, data_s, data_t):
        x_s_1, x_s_2, x_s_3 = self.node_embedding(data_s.x, data_s.edge_index, data_s.batch)
        x_t_1, x_t_2, x_t_3 = self.node_embedding(data_t.x, data_t.edge_index, data_t.batch)

        max_num_nodes = max(max(data_s.__num_nodes_list__), max(data_t.__num_nodes_list__))
        x_1 = self.Maxpadding_and_Resizing(x_t_1, data_t.batch, x_s_1, data_s.batch, max_num_nodes=max_num_nodes)
        x_2 = self.Maxpadding_and_Resizing(x_t_2, data_t.batch, x_s_2, data_s.batch, max_num_nodes=max_num_nodes)
        x_3 = self.Maxpadding_and_Resizing(x_t_3, data_t.batch, x_s_3, data_s.batch, max_num_nodes=max_num_nodes)
        return torch.cat([x_1, x_2, x_3], dim=1)


class CNN(torch.nn.Module):
    def __init__(self, in_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=12, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=1, kernel_size=3, stride=1, padding=(1, 1))

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x


class CLS(nn.Module):
    def __init__(self, hidden_channels, label_size, cnn_channel=3, cnn_reshape=30):
        super(CLS, self).__init__()
        self.cnn = CNN(in_channels=cnn_channel)
        self.linear1 = nn.Linear(cnn_reshape ** 2, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.linear3 = nn.Linear(hidden_channels // 2, label_size)
        self.flatten = nn.Flatten()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.relu()
        x = self.linear3(x)
        x = self.sig(x)
        return x.squeeze(1)


class GraphSim(nn.Module):
    def __init__(self, hidden_channels, label_size, cnn_channel,in_channels,GNNModel,reshape=30):
        super(GraphSim, self).__init__()
        self.gnn = GNN(hidden_channels=hidden_channels, reshape=reshape,in_channels = in_channels,GNNModel = GNNModel)

        self.cls = CLS(hidden_channels=hidden_channels, label_size=label_size,
                       cnn_channel=cnn_channel, cnn_reshape=reshape)

    def forward(self, data1, data2):
        x = self.gnn(data1, data2)
        pred = self.cls(x)
        return pred

    def __repr__(self):
        return 'GraphSim'

