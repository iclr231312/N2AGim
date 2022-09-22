import time
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.datasets import GEDDataset
from torch_geometric.data import DataLoader,Data
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv,GCNConv
from torch_geometric.transforms import OneHotDegree
from torch_geometric.data import Batch
from torchinfo import summary
from torch.nn import init
from scipy.stats import spearmanr, kendalltau
from torch_geometric.utils import softmax, degree, sort_edge_index,to_dense_batch
import json
import os
import argparse
import requests
import prettytable as pt
import datetime
from torch_geometric.utils import degree,to_dense_adj,dense_to_sparse


class DataPair(Data):
    def __init__(self,graph_i,graph_j,ged,**kwargs):
        super(DataPair,self).__init__()
        self["x_i"] =  graph_i.x
        self["x_j"] =  graph_j.x
        self['edge_index_i'] = graph_i.edge_index
        self['edge_index_j'] = graph_j.edge_index
        self['edge_attr_i'] = graph_i.edge_attr
        self['edge_attr_j'] = graph_j.edge_attr
        self["ged"] = ged
        self["pos_x"] = graph_i.i
        self["pos_y"] = graph_j.i
        self['x_i_batch'] = graph_i.batch
        self['x_j_batch'] = graph_j.batch
    def __inc__(self, key, value):
        if key == 'edge_index_i':
            return self.x_i.size(0)
        if key == 'edge_index_j':
            return self.x_j.size(0)
        else:
            return super().__inc__(key, value)

class DegreeSort(object):
    def __init__(self):
        pass
    def __call__(self,data):
        x = data.x
        edge = data.edge_index
        d = degree(edge[0]).numpy()
        idx = np.argsort(-d)
        permute = F.one_hot(torch.from_numpy(idx)).float()
        x = torch.matmul(permute,x)
        adj = to_dense_adj(edge)[0]
        adj = torch.matmul(torch.matmul(permute,adj),permute.transpose(0,1))
        edge = dense_to_sparse(adj)[0]
        data.x = x
        data.edge_index = edge
        # for s in data.edge_stores:
        #     print(s)
        # # print(data.edge_stores)
        # assert 1==0



        return data
        



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

def PrintInfo(config,model_config):
    print('\n=================== Env config INFO  ==========================\n')
    tb =  pt.PrettyTable()
    tb.field_names = ['Hyperparameter','values']
    keys = config
    for key,value in keys.items():
        tb.add_row([key,value])
    tb.align = 'l'
    print(tb)
    print('\n=================== Model config INFO  ==========================\n')
    tb =  pt.PrettyTable()
    tb.field_names = ['Hyperparameter','values']
    keys = model_config
    for key,value in keys.items():
        tb.add_row([key,value])
    tb.align = 'l'
    print(tb)

def PrintInfoDataset(dataset):
    #     print(f'Number of graphs: {len(dataset)}')
   
    num_node,num_edge = 0,0
    max_node,max_edge = 0,0
    min_node,min_edge = dataset[0].num_nodes,dataset[0].num_edges
    for data in dataset:
        max_node,max_edge = max(max_node,data.num_nodes),max(max_edge,data.num_edges)
        num_node = num_node + data.num_nodes
        num_edge = num_edge + data.num_edges
        min_node,min_edge = min(min_node,data.num_nodes),min(min_edge,data.num_edges)

    print(data)
    print(f"Num of graphs:{len(dataset)}")
    print(f'Avg of nodes: {num_node / len(dataset),max_node,min_node}')
    print(f'Number of edges: {num_edge / len(dataset),max_edge,min_edge}')
    print(f'Average node degree: {num_edge/ num_node:.2f}')

# def PrintInfo(train_datasets,vaild_datasets,test_datasets,config):
#     dataset = train_datasets+vaild_datasets+test_datasets
#     print(f'==============  Dataset INFO: {config["datasets"]}  ===========================\n')
#     print(f'dataset spilt -> train / vaild / test : {len(train_datasets)} / {len(vaild_datasets)} / {len(test_datasets)}')
#     print(f'Number of graphs: {len(dataset)}')
   
#     data = train_datasets[0]  # Get the first graph object.
#     print(data)
#     # Gather some statistics about the first graph.
#     print(f'Number of features: {data.num_features}')
#     print(f'Number of nodes: {data.num_nodes}')
#     print(f'Number of edges: {data.num_edges}')
#     print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
#     print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
#     print(f'Contains self-loops: {data.contains_self_loops()}')
#     print(f'Is undirected: {data.is_undirected()}')
#     print('\n===================  config INFO  =================================\n')
#     tb =  pt.PrettyTable()

#     tb.field_names = ['Hyperparameter','values']
#     keys = config
#     for key,value in keys.items():
#         tb.add_row([key,value])
#     tb.align = 'l'
#     print(tb)

class MyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        if isinstance(obj,datetime.timedelta):
            return str(obj, encoding='utf-8')
        if isinstance(obj,np.float32):
            return float(obj)
        if isinstance(obj,np.integer):
            return int(obj)
        if isinstance(obj,np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def dict2json(file_name,the_dict):

    json_str = json.dumps(the_dict,indent=4,cls = MyEncoder)
    with open(file_name, 'w') as json_file:
        json_file.write(json_str)



def ranking_func(data):
    sort_id_mat = np.argsort(-data)
    n = sort_id_mat.shape[0]
    rank = np.zeros(n)
    for i in range(n):
        finds = np.where(sort_id_mat == i)
        fid = finds[0][0]
        while fid > 0:
            cid = sort_id_mat[fid]
            pid = sort_id_mat[fid - 1]
            if data[pid] == data[cid]:
                fid -= 1
            else:
                break
        rank[i] = fid + 1

    return rank


def calculate_ranking_correlation(rank_corr_function, prediction, target):
    """
    Calculating specific ranking correlation for predicted values.
    :param rank_corr_function: Ranking correlation function.
    :param prediction: Vector of predicted values.
    :param target: Vector of ground-truth values.
    :return ranking: Ranking correlation value.
    """
    r_prediction = ranking_func(prediction)
    r_target = ranking_func(target)
    return rank_corr_function(r_prediction, r_target).correlation


def top_k_ids(data, k, inclusive, rm):
    """
    :param data: input
    :param k:
    :param inclusive: whether to be tie inclusive or not.
        For example, the ranking may look like this:
        7 (sim_score=0.99), 5 (sim_score=0.99), 10 (sim_score=0.98), ...
        If tie inclusive, the top 1 results are [7, 9].
        Therefore, the number of returned results may be larger than k.
        In summary,
            len(rtn) == k if not tie inclusive;
            len(rtn) >= k if tie inclusive.
    :param rm: 0
    :return: for a query, the ids of the top k database graph
    ranked by this model.
    """
    sort_id_mat = np.argsort(-data)
    n = sort_id_mat.shape[0]
    if k < 0 or k >= n:
        raise RuntimeError('Invalid k {}'.format(k))
    if not inclusive:
        return sort_id_mat[:k]
    # Tie inclusive.
    dist_sim_mat = data
    while k < n:
        cid = sort_id_mat[k - 1]
        nid = sort_id_mat[k]
        if abs(dist_sim_mat[cid] - dist_sim_mat[nid]) <= rm:
            k += 1
        else:
            break
    return sort_id_mat[:k]


def prec_at_ks(true_r, pred_r, ks, rm=0):
    """
    Ranking-based. prec@ks.
    :param true_r: result object indicating the ground truth.
    :param pred_r: result object indicating the prediction.
    :param ks: k
    :param rm: 0
    :return: precision at ks.
    """
    true_ids = top_k_ids(true_r, ks, inclusive=True, rm=rm)
    pred_ids = top_k_ids(pred_r, ks, inclusive=True, rm=rm)
    ps = min(len(set(true_ids).intersection(set(pred_ids))), ks) / ks
    return ps


def metric(p_mat,gt,verbose):
    rho_list = []
    tau_list = []
    prec_at_10_list = []
    prec_at_20_list = []
    par = tqdm(range(len(p_mat))) if verbose else range(len(p_mat))
    for i in par:
        rho_list.append(calculate_ranking_correlation(spearmanr, p_mat[i], gt[i]))
        tau_list.append(calculate_ranking_correlation(kendalltau, p_mat[i], gt[i]))
        prec_at_10_list.append(prec_at_ks(gt[i], p_mat[i], 10))
        prec_at_20_list.append(prec_at_ks(gt[i], p_mat[i], 20))
    return rho_list,tau_list,prec_at_10_list,prec_at_20_list

def myproduct(*args, repeat=1):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield list(prod)


def cut(obj, sec):
    return [obj[i:i + sec] for i in range(0, len(obj), sec)]


def list2loader(data_list, num_cut):
    data_list = cut(data_list, num_cut)
    for i in data_list:
        s = [i[idx][0] for idx in range(len(i))]
        t = [i[idx][1] for idx in range(len(i))]
        s = Batch.from_data_list(s)
        t = Batch.from_data_list(t)
        yield s, t

# if __name__ == "__main__":
