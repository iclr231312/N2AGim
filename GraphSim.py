import torch
from torch import conv2d, nn
from torch_geometric.nn import SAGEConv,GCNConv
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_sum
import copy
from utils import DataPair

class GNNLayers(torch.nn.Module):
    def __init__(self, in_channels,reshape):
        super(GNNLayers, self).__init__()
        self.gcn1 = GCNConv(in_channels, 64)
        self.gcn2 = GCNConv(64, 32)
        self.gcn3 = GCNConv(32, 16)
        self.reshape = reshape
        self.activate = [nn.ReLU() for i in range(3)]

    def Maxpadding_and_Resizing(self, x_t, batch_t, x_s, batch_s,max_num_nodes):
        x_t = to_dense_batch(x_t, batch=batch_t, max_num_nodes=max_num_nodes)[0]
        x_s = to_dense_batch(x_s, batch=batch_s, max_num_nodes=max_num_nodes)[0]

        x = torch.bmm(x_s, x_t.transpose(1, 2))
        x = nn.functional.interpolate(x.unsqueeze(1), size=(self.reshape,self.reshape), mode='bilinear', align_corners=False)
        return x

    def node_embedding(self, x, edge_index, batch):
        x = self.gcn1(x, edge_index).relu()
        # x = self.activate[0](x)
        x1 = x.clone()
        x = self.gcn2(x, edge_index).relu()
        # x = self.activate[1](x)
        x2 = x.clone()
        x = self.gcn3(x, edge_index).relu()
        # x = self.activate[2](x)
        x3 = x.clone()

        return x1, x2, x3

    def forward(self, pair):
        x_s_1, x_s_2, x_s_3 = self.node_embedding(pair.x_i, pair.edge_index_i, pair.x_i_batch)
        x_t_1, x_t_2, x_t_3 = self.node_embedding(pair.x_j, pair.edge_index_j, pair.x_j_batch)

        max_node_i = scatter_sum(torch.ones_like(pair.x_i_batch),pair.x_i_batch).max().item()
        max_node_j = scatter_sum(torch.ones_like(pair.x_j_batch),pair.x_j_batch).max().item()
        max_num_nodes = max(max_node_i, max_node_j)

        x_1 = self.Maxpadding_and_Resizing(x_t_1, pair.x_j_batch, x_s_1, pair.x_i_batch, max_num_nodes=max_num_nodes)
        x_2 = self.Maxpadding_and_Resizing(x_t_2, pair.x_j_batch, x_s_2, pair.x_i_batch, max_num_nodes=max_num_nodes)
        x_3 = self.Maxpadding_and_Resizing(x_t_3, pair.x_j_batch, x_s_3, pair.x_i_batch, max_num_nodes=max_num_nodes)
        return x_1, x_2, x_3


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=6,stride=1),
            # nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=6,stride=1),
            # nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1),
            # nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1),
            # nn.MaxPool2d(3)
            # nn.ReLU()
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
    def forward(self, x):
        x = self.cnn(x)
        return x


class CLS(nn.Module):
    def __init__(self, hidden_size=32):
        super(CLS, self).__init__()

        self.cnn = nn.ModuleList([CNN(),CNN(),CNN()])
        self.linear = nn.Sequential(
                # nn.Flatten(),
                nn.Linear(32 * 3, hidden_size//2),
                nn.ReLU(),
                nn.Linear(hidden_size//2, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
        )

    def forward(self, x_1, x_2, x_3):
        x_1 = self.cnn[0](x_1)
        x_2 = self.cnn[1](x_2)
        x_3 = self.cnn[2](x_3)
        x = torch.cat([x_1,x_2,x_3],dim=1)
        x = self.linear(x)
        return x.squeeze(1)


class graphsim(nn.Module):
    def __init__(self, in_channels,args):

        super(graphsim, self).__init__()
        self.max_num_nodes = 0
        self.gnn = GNNLayers(in_channels = in_channels,reshape=30)
        self.cls = CLS(hidden_size = 32)

    def forward(self,graph_i,graph_j):
        pair = DataPair(graph_i,graph_j,ged = 0)
        x_1, x_2, x_3 = self.gnn(pair)

        pred = self.cls(x_1, x_2, x_3)
        return pred

    def __repr__(self):
        return 'GraphSim'

