from operator import truediv
from tqdm import tqdm
import torch
import numpy as np
from torch import nn
from torch_geometric.nn import global_mean_pool,global_add_pool
from torch_geometric.utils import degree,to_dense_batch
from layers import *
from torch_scatter import scatter_sum
from utils import DataPair

class GCN_GNNModel(nn.Module):
    """
    GNN model,
    input : x, edge_index, batch
    output: list of [batch, node, channels]
    """
    def __init__(self, in_channels, out_channels, gnn_num, return_mode = 'dense'):
        super(GCN_GNNModel, self).__init__()

        assert return_mode in ['dense','sparse']
        self.return_mode = return_mode

        self.gnn = []
        self.first_head = GCNConv(in_channels,out_channels)
        self.bn = [nn.BatchNorm1d(out_channels)]
        for i in range(gnn_num):
            self.gnn.append(SAGEConv(out_channels,out_channels))
            self.bn.append(nn.BatchNorm1d(out_channels,affine = False))
        self.gnn = nn.ModuleList(self.gnn)
        self.bn = nn.ModuleList(self.bn)

    def forward(self,x, edge_index, batch):

        #first embedding the input graph
        x = self.first_head(x, edge_index).relu()
        # x = self.bn[0](x)

        if self.return_mode == 'dense':
            var_temp = [to_dense_batch(x, batch)[0]]
            #apply to gnn
            for idx in range(len(self.gnn)):

                x = self.gnn[idx](x,edge_index).relu()
                # x = self.bn[idx+1](x)
                var_temp.append(to_dense_batch(x, batch)[0])


        elif self.return_mode == 'sparse':
            var_temp = [x]

            #apply to gnn
            for idx in range(len(self.gnn)):

                x = self.gnn[idx](x,edge_index).relu()
                # x = self.bn[idx+1](x)
                var_temp.append(x)


        return var_temp

class simgnn(nn.Module):
    def __init__(self,in_channels,args) -> None:
        super(simgnn,self).__init__()
        self.params = {
            "out_channels" : 1,
            "hidden_channels": 32,
            "k":32,
            "gnn_num" : 2,
            "mode" : "sparse",
            "bin" : 16,
            "cal_hist" : False
        }
        self.GNN = GCN_GNNModel(in_channels, self.params["hidden_channels"], self.params["gnn_num"], return_mode = self.params["mode"])

        self.weight_c = torch.nn.Parameter(torch.FloatTensor(self.params["hidden_channels"],self.params["hidden_channels"]))

        self.W = torch.nn.Parameter(torch.FloatTensor(self.params["k"],self.params["hidden_channels"],self.params["hidden_channels"]))
        self.V = torch.nn.Parameter(torch.FloatTensor(self.params["k"],2*self.params["hidden_channels"]))
        self.bias = torch.nn.Parameter(torch.FloatTensor(self.params["k"]))

        # self.mlp = nn.Sequential(
        #     nn.Linear(self.params["k"] + self.params["bin"] if self.params["cal_hist"] else self.params["k"],self.params['hidden_channels'] //2),
        #     # nn.BatchNorm1d(self.params['hidden_channels'] //2),
        #     nn.ReLU(),
        #     nn.Linear(self.params['hidden_channels'] //2,self.params['out_channels'])
        #     # nn.Sigmoid()
        # )
        self.mlp = nn.Sequential(
            nn.Linear(self.params["k"] + self.params["bin"] if self.params["cal_hist"] else self.params["k"],self.params['out_channels']),
            # # nn.BatchNorm1d(self.params['hidden_channels'] //2),
            # nn.ReLU(),
            # nn.Linear(self.params['hidden_channels'] //2,self.params['out_channels'])
            # nn.Sigmoid()
        )
        self.reset_parameters()
    def forward(self,graph_i,graph_j):
        # Stage I: Node Embedding
        pair = DataPair(graph_i,graph_j,ged = 0)
        graph_i_representation = self.GNN(pair.x_i, pair.edge_index_i, pair.x_i_batch)[-1]
        graph_j_representation = self.GNN(pair.x_j, pair.edge_index_j, pair.x_j_batch)[-1]

        #Stage II: Graph Embedding: Global Context-Aware Attention.

        c_i = global_mean_pool(graph_i_representation,pair.x_i_batch)
        c_j = global_mean_pool(graph_j_representation,pair.x_j_batch)

        c_i = torch.tanh(torch.matmul(c_i,self.weight_c))
        c_j = torch.tanh(torch.matmul(c_j,self.weight_c))

        c_i = c_i[pair.x_i_batch]
        c_j = c_j[pair.x_j_batch]

        h_i = global_add_pool( torch.mul(torch.sigmoid(torch.sum(graph_i_representation * c_i,dim = 1)).unsqueeze(1),graph_i_representation),pair.x_i_batch)
        h_j = global_add_pool( torch.mul(torch.sigmoid(torch.sum(graph_j_representation * c_j,dim = 1)).unsqueeze(1),graph_j_representation),pair.x_j_batch)

        #Stage III: Graph-Graph Interaction: Neural Tensor Network.

        h_ij = torch.cat([h_i,h_j],dim = 1)

        h_i = h_i.unsqueeze(1).unsqueeze(1)
        h_i = torch.cat([h_i for i in range(self.params['k'])],dim = 1) # batch,k,1,hidden_channels

        h_j = h_j.unsqueeze(2).unsqueeze(1)
        h_j = torch.cat([h_j for i in range(self.params['k'])],dim = 1) # batch,k,hidden_channels,1

        g = torch.matmul(torch.matmul(h_i,self.W),h_j).squeeze() + torch.matmul(h_ij,self.V.t()).squeeze() + self.bias

        # calculate_histogram

        if self.params['cal_hist']:
                hist = self.calculate_histogram(graph_i_representation, graph_j_representation, pair)
                # print(g.dtype,hist.dtype)
                g = torch.cat([g,hist],dim = 1)

        # mlp

        scores = self.mlp(g).squeeze()
        # print(scores)
        return scores

    def calculate_histogram(self, h_i, h_j, pair):
        """
        https://github.com/benedekrozemberczki/SimGNN/blob/master/src/simgnn.py#L50
        """
        max_node_i = scatter_sum(torch.ones_like(pair.x_i_batch),pair.x_i_batch).max().item()
        max_node_j = scatter_sum(torch.ones_like(pair.x_j_batch),pair.x_j_batch).max().item()
        max_num_nodes = max(max_node_i, max_node_j)
        node_level_representation_i = to_dense_batch(h_i,pair.x_i_batch,max_num_nodes=max_num_nodes)[0] # shape [batch,max_node,channels]
        node_level_representation_j = to_dense_batch(h_j,pair.x_j_batch,max_num_nodes=max_num_nodes)[0] # shape [batch,max_node,channels]
        batch_size = node_level_representation_j.shape[0]

        scores = torch.matmul(node_level_representation_i,node_level_representation_j.transpose(1,2)).detach() # shape [batch,max_node,max_node]

        scores = scores.reshape(batch_size, -1)

        hist = [torch.histc(score, bins=self.params['bin']).tolist() for score in scores]
        hist = torch.from_numpy(np.array(hist))

        hist = hist/torch.sum(hist,dim = 1).unsqueeze(1)
        hist = hist.to(node_level_representation_i.device).type(node_level_representation_i.dtype)

        return hist

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.weight_c)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.V)
        # nn.init.xavier_uniform_(self.bias)

    def __repr__(self) -> str:
        return "SimGNN"