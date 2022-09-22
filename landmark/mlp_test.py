"""
This script can be leveraged to test the accuracy and the inference time of our GSL^2
"""
import torch
import os
import torch_geometric
import random
import numpy as np
import matplotlib.pylab as pl
# from torch_geometric.data import pygBatch
from torch_geometric.utils import to_dense_batch, to_networkx,degree
from torch_geometric.datasets import GEDDataset
from torch_geometric.data import DataLoader as pygDataLoader
import pprint
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import pandas as pd
from tqdm import tqdm
import argparse
import sys
import time
from torch import nn
# import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
os.chdir("../")
sys.path.append("./")
from train import *
from utils import *
import logging
# import logging
############# process function #####################

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='AIDS700nef',
                    help="IMDBMulti,AIDS700nef,LINUX")
parser.add_argument("--epochs", type=int, default=500,
               help="epochs")
# parser.add_argument("--name", type=str, default="full")
parser.add_argument("--size", type=int, default=20,
               help="how many graphs are random selected.")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch", type=int, default=5000, help="batch size")
parser.add_argument("--gpu", type=str, default='0',
               help='defined which gpu should be used')
parser.add_argument("--model", type=str, default="N2Gim",
               help='defined which graph similarity model should be used')
# parser.add_argument("--work_dir",type=str,help='the files including the model paramters.')
# parser.add_argument("--model",type=str,default='xgboost', help='et,knn,rf,catboost,xgboost,lr,br')
# parser.add_argument("--graph_level_type", type=str, default='gaddp')
args = parser.parse_args()
# args.work_dir = "./landmarkv4/"+args.work_dir
# assert args.st in ["random", "traverse"]


CHANNELS_LENGTH = args.size


def transform2GT(x, n):
   # n = n.to(x.device)
   # print(x.shape)
   # print(n.shape)
   x = x.cpu()
   return torch.exp(-x/n.unsqueeze(1))


def num_nodes(graph_batch):
    return graph_batch.ptr[1:] - graph_batch.ptr[:-1]


def mysimilarity2GED(x, n1, n2, max_v):
   #  max_v = torch.Tensor([32.2292]).to(x.device)
    max_v = torch.Tensor([max_v]).to(x.device)
    norm_ged = 10**((1 - x)*torch.log10(max_v + 1)) - 1
    return ((n1+n2)/2)*norm_ged  # ((n1+n2)/2)*


def similarity2GED(x, n1, n2):
    return -((n1+n2)/2)*torch.log(x)*10


def GetKernel(datasets_1, datasets_2, model, max_v):
    prediction_mat = []
    groundtrue_mat = []
    ged = datasets_1.ged
    device = next(model.parameters()).device
    data_all = list(myproduct(datasets_1, datasets_2))
    dl = pygDataLoader(data_all, shuffle=False, batch_size=2000, follow_batch=[
                       'x_i', 'x_j'], num_workers=7, pin_memory=True)
    pred = []
    model.eval()
    with torch.no_grad():
      #   pbar = tqdm(dl)
        pbar = tqdm(dl,total = len(dl))
        for s, t in pbar:
            s = s.to(device)
            t = t.to(device)
            pred = model(s, t)
            label = ged[s.i, t.i].to(device)
            num_nodes_s = num_nodes(s)
            num_nodes_t = num_nodes(t)

            # prediction_mat.append(pred)
            # groundtrue_mat.append(label)
            prediction_mat.append(mysimilarity2GED(
                pred, num_nodes_s, num_nodes_t, max_v=max_v))
            groundtrue_mat.append(label)  # label
            # print(similarity2GED(label,num_nodes_s,num_nodes_t), train_datasets.ged[s.i,t.i])

    prediction_mat = torch.cat(prediction_mat, dim=0)
    groundtrue_mat = torch.cat(groundtrue_mat, dim=0)
    p_mat = prediction_mat.reshape(len(datasets_1), -1).T.cpu().numpy()
    gt = groundtrue_mat.reshape(len(datasets_1), -1).T.cpu().numpy()
    return p_mat, gt


def similarity(gt, threshold, Id_select, node_id):
   node_id = [node_id]*len(Id_select)
   similarity_metric = gt[Id_select, node_id]
   return True if np.all(similarity_metric <= threshold) else False


# def getLandmark(gt, threshold):
#    assert gt.shape[0] == gt.shape[1]
#    Id = gt.shape[0]
#    Id = np.array(list(range(Id)))
#    Id_select = [0]
#    if datasets == "LINUX":
#       np.random.seed(8)
#       np.random.shuffle(Id)
#    for i in Id:
#       if similarity(gt, threshold, Id_select, i):
#          Id_select.append(i)
#    return Id_select


def getLandmarkRandom(gt, size):
   st_size = size
   Id_select = np.random.choice(gt.shape[0], st_size).tolist()
   return Id_select


def process_features(xi, xj):
   return np.concatenate([xi, xj], axis=0).tolist()
   # return np.concatenate([np.abs(xi - xj),xj + xi],axis = 0).tolist()


def getnodelist(dataset):
   node_list = []
   for g in dataset:
      # print(g.num_nodes)
      node_list.append(g.num_nodes)
      # print(node_list)
      # assert 1 == 0
   return node_list


class data_container():
   def __init__(self, args):
      super().__init__()
      self.args = args
      torch.manual_seed(2)
      np.random.seed(2)
      datasets = args.dataset
      train_datasets_all = GEDDataset(
          "../SomeData4kaggle/{}".format(datasets), name=datasets)
      self.test_datasets = GEDDataset(
          "../SomeData4kaggle/{}".format(datasets), name=datasets, train=False)
    #   device = torch.device('cuda:{}'.format(args.gpu)
    #                         if torch.cuda.is_available() else 'cpu')
      device = torch.device('cpu')
    #   device = torch.device('cpu')
      max_degree = 0
      for g in train_datasets_all + self.test_datasets:
         if g.edge_index.size(1) > 0:
            max_degree = max(max_degree, int(
                degree(g.edge_index[0]).max().item()))
      one_hot_degree = OneHotDegree(max_degree, cat=True)
      train_datasets_all.transform = one_hot_degree
      self.test_datasets.transform = one_hot_degree
      train_datasets_all = train_datasets_all.shuffle()
      idx = int(len(train_datasets_all)*0.75)
      indices = np.random.permutation(len(train_datasets_all))
      self.train_datasets, self.vaild_datasets = train_datasets_all[indices[:idx].tolist(
      )], train_datasets_all[indices[idx:].tolist()]

      # .format(datasetsname, args.model,args.graph_level_type)
      save_file_name = f"./result_norm/{datasets}/{args.model}/sum_none_pGinFalse_woNodeattFalse_woGraphattFalse/minvaild_0.pt"
      self.model = torch.load(save_file_name).to(device)
      with open(f'./result_norm/{datasets}/{args.model}/sum_none_pGinFalse_woNodeattFalse_woGraphattFalse/loss.json', 'r') as f:
         self.max_v = json.load(f)["max"]
      self.landmark_id = np.load(
          f"./landmark/save/{datasets}_{args.size}_landmarkid.npy")
      p_mat_train, gt_train = GetKernel(
          self.train_datasets, self.train_datasets, self.model, self.max_v)
      p_mat_val, gt_val = GetKernel(
          self.train_datasets, self.vaild_datasets, self.model, self.max_v)
      p_mat_test, gt_test = GetKernel(
          self.train_datasets, self.test_datasets, self.model, self.max_v)

      train_nodes, val_nodes, test_nodes = getnodelist(self.train_datasets), getnodelist(
          self.vaild_datasets), getnodelist(self.test_datasets)
      self.node_dict = dict(train=train_nodes, val=val_nodes,
                            test=test_nodes, test_graphs_size=len(self.test_datasets))

      self.gt = dict(train=gt_train, val=gt_val, test=gt_test)
      self.data = dict(train=self.train_datasets,
                       val=self.vaild_datasets, test=self.test_datasets)
      self.landmark_features_train, _ = GetKernel(
          self.train_datasets[self.landmark_id], self.train_datasets, self.model, self.max_v)
      self.length_size = len(self.train_datasets)
   def get_datasets(self, data_spilt_name='test'):
      gt_data = self.gt[data_spilt_name]

      landmark_features_data, _ = GetKernel(
          self.train_datasets[self.landmark_id], self.data[data_spilt_name], self.model, self.max_v)
      data_pair = []
      data_target = []
      # print(np.array(landmark_features_data).shape)
      # print(gt_data.shape)
      for ids, i in enumerate(landmark_features_data):
         for idx, j in enumerate(self.landmark_features_train):
            # print(i)
            data_pair.append(process_features(i, j))
            data_target.append(gt_data[ids, idx])
      data_pair = torch.from_numpy(np.array(data_pair))
      data_target = torch.from_numpy(np.array(data_target))
      data_tensordatset = TensorDataset(data_pair, data_target)
      return data_tensordatset


def data_process():
   torch.manual_seed(2)
   np.random.seed(2)
   datasets = args.dataset
   datasetsname = datasets
   model_name = args.model
   gpu = args.gpu
   mse_arr = []
   rho_arr = []
   tau_arr = []
   prec_at_10_arr = []
   prec_at_20_arr = []
   
   train_datasets_all = GEDDataset(
       "../SomeData4kaggle/{}".format(datasetsname), name=datasetsname)
   test_datasets = GEDDataset(
       "../SomeData4kaggle/{}".format(datasetsname), name=datasetsname, train=False)
   max_degree = 0
   for g in train_datasets_all + test_datasets:
      if g.edge_index.size(1) > 0:
         max_degree = max(max_degree, int(
             degree(g.edge_index[0]).max().item()))
   one_hot_degree = OneHotDegree(max_degree, cat=True)
   train_datasets_all.transform = one_hot_degree
   test_datasets.transform = one_hot_degree
   train_datasets_all = train_datasets_all.shuffle()
   idx = int(len(train_datasets_all)*0.75)
   indices = np.random.permutation(len(train_datasets_all))
   train_datasets, vaild_datasets = train_datasets_all[indices[:idx].tolist(
   )], train_datasets_all[indices[idx:].tolist()]

   # for i in range(20):
   #    print(vaild_datasets[i].i)

   # .format(datasetsname, args.model,args.graph_level_type)
   save_file_name = f"./result_norm/{datasets}/{args.model}/sum_so_pGinFalse_woattFalse/minvaild_0.pt"
   model = torch.load(save_file_name).to(device)
   with open(f'./result_norm/{datasets}/{args.model}/sum_so_pGinFalse_woattFalse/loss.json', 'r') as f:
      max_v = json.load(f)["max"]
   # if not os.path.exists(f"./landmarkv4/temp/{datasets}_p_mat_train.pt"):
   p_mat_train, gt_train = GetKernel(
       train_datasets, train_datasets, model, max_v)
   p_mat_val, gt_val = GetKernel(train_datasets, vaild_datasets, model, max_v)
   p_mat_test, gt_test = GetKernel(train_datasets, test_datasets, model, max_v)

   del model
   torch.cuda.empty_cache()
   landmark_id = torch.load(
       f"./landmark/save/{datasets}_{args.size}_landmarkid.npy")
   CHANNELS_LENGTH = len(landmark_id)
   landmark_features_train = p_mat_train[:, landmark_id]

   train_pair = []
   train_target = []
   for ids, i in enumerate(landmark_features_train):
      for idx, j in enumerate(landmark_features_train):
         train_pair.append(process_features(i, j))
         train_target.append(gt_train[ids, idx])
   train_pair = torch.from_numpy(np.array(train_pair))
   train_target = torch.from_numpy(np.array(train_target))
   train_tensordatset = TensorDataset(train_pair, train_target)

   landmark_features_val = p_mat_val[:, landmark_id]
   val_pair = []
   val_target = []
   for ids, i in enumerate(landmark_features_val):
      for idx, j in enumerate(landmark_features_train):
         val_pair.append(process_features(i, j))
         val_target.append(gt_val[ids, idx])

   val_pair = torch.from_numpy(np.array(val_pair))
   val_target = torch.from_numpy(np.array(val_target))
   val_tensordatset = TensorDataset(val_pair, val_target)

   landmark_features_test = p_mat_test[:, landmark_id]
   test_pair = []
   test_target = []
   for ids, i in enumerate(landmark_features_test):
      for idx, j in enumerate(landmark_features_train):
         test_pair.append(process_features(i, j))
         test_target.append(gt_test[ids, idx])

   test_pair = torch.from_numpy(np.array(test_pair))
   test_target = torch.from_numpy(np.array(test_target))
   test_tensordatset = TensorDataset(test_pair, test_target)

   train_nodes, val_nodes, test_nodes = getnodelist(
       train_datasets), getnodelist(vaild_datasets), getnodelist(test_datasets)
   node_dict = dict(train=train_nodes, val=val_nodes,
                    test=test_nodes, test_graphs_size=len(test_datasets))
   return train_tensordatset, val_tensordatset, test_tensordatset, node_dict
####################################


class MLP(nn.Module):
   def __init__(self, input_channels, hidden_channels, num_mlp):
      super(MLP, self).__init__()
      self.mlp_head = nn.Sequential(
         nn.Linear(input_channels, hidden_channels),
         nn.ReLU()
         )
      mlp_block = nn.Sequential(
         nn.Linear(hidden_channels, hidden_channels),
         nn.ReLU()
         )
      self.mlp_tail = nn.Sequential(
         nn.Linear(hidden_channels, 1),
         )

      self.mlp_blocks = nn.ModuleList([mlp_block for _ in range(num_mlp)])

   def forward(self, x):
      x = self.mlp_head(x)
      for mlp_block in self.mlp_blocks:
         x = x + mlp_block(x)
      return self.mlp_tail(x)


def vaild_step(net, loader, node_dict):
   net.eval()
   loss_arr = []
   node_lists = []
   for i in node_dict.get("val"):
      for j in node_dict.get("train"):
         node_lists.append((i+j)/2)
   node_lists = torch.Tensor(node_lists)
   predition = []
   target = []
   with torch.no_grad():
      pbar = loader#pbar = tqdm(loader)
      for idx, (x, y) in enumerate(pbar):
         x = x.to(device)
         y = y.to(device).double().unsqueeze(1)
         y_pred = net(x)
         predition.append(y_pred)
         target.append(y)
      predition = transform2GT(torch.cat(predition, dim=0), node_lists)
      target = transform2GT(torch.cat(target, dim=0), node_lists)
      loss_arr = F.mse_loss(predition, target)
   net.train()
   return torch.mean(loss_arr)


def test(net, dc,num_rounds = 5, need_print = False):
   inference_R = []
   inference_F = []
#    device = 'cpu'
#    net.to(device)
   node_dict = dc.node_dict
   net.eval()
   node_lists = []
   for i in node_dict.get("test"):
      for j in node_dict.get("train"):
         node_lists.append((i+j)/2)
   node_lists = torch.Tensor(node_lists)

   for _ in range(1):
      predition = []
      target = []
      # torch.cuda.synchronize()
      # start = time.time()
      test_dataset = dc.get_datasets("test")
    #   x, y = test_dataset.tensors
      loader = DataLoader(test_dataset, shuffle=False, batch_size=2000, follow_batch=[
                       'x_i', 'x_j'], num_workers=7, pin_memory=True)
      with torch.no_grad():
         torch.cuda.synchronize()
         off_line_start = time.time()
         for x,y in tqdm(loader,total = len(loader)):
            x = x.to(device)
            y = y.to(device).double().unsqueeze(1)
            y_pred = net(x)
            predition.append(y_pred)
            target.append(y)
        #  with torch.autograd.profiler.profile(use_cuda=True) as prof:
         # torch.cuda.synchronize()
         # end = time.time()
         # inference_R.append(end-start)
         # inference_F.append(end-off_line_start)
         predition = transform2GT(torch.cat(predition,dim = 0),node_lists).numpy().reshape(node_dict["test_graphs_size"],-1)
         target = transform2GT(torch.cat(target,dim = 0),node_lists).numpy().reshape(node_dict["test_graphs_size"],-1)
         mse = ((predition - target)**2).mean()

         # tqdm.write("test off-line time:", end - start, " second")
   
   rho_list, tau_list, prec_at_10_list, prec_at_20_list = metric(predition, target,verbose = False)
   rho,tau,prec_at_10,prec_at_20 = np.mean(rho_list), np.mean(tau_list), np.mean(prec_at_10_list), np.mean(prec_at_20_list)

   net.train()
   # if need_print:
   #  print("inference_R",inference_R)
   #  print("mean inference_R",np.mean(inference_R))
   #  print("inference_F",inference_F)
   #  print("mean inference_F",np.mean(inference_F))
#    assert 1 == 0
   return rho,tau,prec_at_10,prec_at_20,mse

if __name__ == '__main__':
   tqdm.write("------------------------------------------------")
   tqdm.write(f"{args.dataset}_{args.size}")
   dc = data_container(args)
   # train_tensordatset,val_tensordatset,test_tensordatset,node_dict = data_process()
   # node_dict = dc.node_dict
   # net = MLP(input_channels=2*CHANNELS_LENGTH,hidden_channels=128,num_mlp=3).to(device)
   # net = net.double()
   # optim = torch.optim.Adam(lr=args.lr, params=net.parameters())

   # train_loader = DataLoader(dataset=train_tensordatset,shuffle=True,batch_size=args.batch)
   # val_loader = DataLoader(dataset=val_tensordatset,shuffle=False,batch_size=args.batch)
   # test_loader = DataLoader(dataset=test_tensordatset,shuffle=False,batch_size=args.batch)
#    device = torch.device('cuda:{}'.format(args.gpu)
#                             if torch.cuda.is_available() else 'cpu')
   device = torch.device(f'cuda:{args.gpu}')
   net = torch.load(f"landmark/results/{args.dataset}_{args.size}/{args.dataset}_{args.size}_minvaild.pt").to(device)
   # warm up
   rho,tau,prec_at_10,prec_at_20,mse = test(net,dc,num_rounds = 1,need_print = False)
   # rho,tau,prec_at_10,prec_at_20,mse = test(net,dc,num_rounds = 5,need_print = True)

   tqdm.write("mse is {}  10^-3".format(mse*1000))
   tqdm.write("rho is {}".format(rho))
   tqdm.write("tau is {}".format(tau))
   tqdm.write("p@10 is {}".format(prec_at_10))
   tqdm.write("p@20 is {}".format(prec_at_20))
   tqdm.write("------------------------------------------------")