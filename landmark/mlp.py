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
from tqdm.notebook import tqdm
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
############# process function #####################

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='AIDS700nef', help="IMDBMulti,AIDS700nef,LINUX")
parser.add_argument("--epochs", type=int, default=500,
               help="epochs")
# parser.add_argument("--name", type=str, default="full")
parser.add_argument("--size", type=int, default=20,
               help="how many graphs are random selected.")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch", type=int,default=5000, help="batch size")
parser.add_argument("--gpu", type=int, default=3,
               help='defined which gpu should be used')
parser.add_argument("--model", type=str, default="N2Gim",
               help='defined which graph similarity model should be used')
# parser.add_argument("--pca",type=int,default=7)
# parser.add_argument("--model",type=str,default='xgboost', help='et,knn,rf,catboost,xgboost,lr,br')
# parser.add_argument("--graph_level_type", type=str, default='gaddp')
args = parser.parse_args()

# assert args.st in ["random", "traverse"]



CHANNELS_LENGTH = args.size


def getLogger(dataset_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    # FileHandler
    work_dir = f"./landmark/results/{dataset_name}_{args.size}"
   #  work_dir = os.path.join(f"./landmark/results/{dataset_name}_{args.size}",
   #                          time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime()))  # 日志文件写入目录
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
    fHandler.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    fHandler.setFormatter(formatter)  # 定义handler的输出格式
    logger.addHandler(fHandler)  # 将logger添加到handler里面

    return logger,work_dir


def transform2GT(x,n):
   x = x.cpu()
   return torch.exp(-x/n.unsqueeze(1))

def num_nodes(graph_batch):
    return graph_batch.ptr[1:] - graph_batch.ptr[:-1]

def mysimilarity2GED(x,n1,n2,max_v):
   #  max_v = torch.Tensor([32.2292]).to(x.device)
    max_v = torch.Tensor([max_v]).to(x.device)
    norm_ged = 10**((1 - x)*torch.log10(max_v + 1)) - 1
    return ((n1+n2)/2)*norm_ged#((n1+n2)/2)*

def similarity2GED(x,n1,n2):
    return -((n1+n2)/2)*torch.log(x)*10

def GetKernel(datasets_1,datasets_2,model,max_v):
    prediction_mat = []
    groundtrue_mat = []
    ged = datasets_1.ged
    # ged = torch.exp(-ged)
   #  print(ged.max(),ged.min())
    #assert 1==0
    device = next(model.parameters()).device
    data_all = list(myproduct(datasets_1,datasets_2))
    dl = pygDataLoader(data_all,shuffle=False,batch_size=2000,follow_batch = ['x_i','x_j'], num_workers=7,pin_memory = True)
#     del pair
    pred = []
    model.eval()
    with torch.no_grad():
        pbar = tqdm(dl)
        for s,t in pbar:
            # assert 1 == 0
            s = s.to(device)
            t = t.to(device)
            pred = model(s,t)
            # pred = nn.functional.relu(pred)+1e-14
            # print("pred : ",pred.max(),pred.min())
            label = ged[s.i,t.i].to(device)
            # print("label : ",label.max(),label.min())
            # print(pred.shape)
            # assert 1 == 0
            # torch.log()
            num_nodes_s = num_nodes(s)
            num_nodes_t = num_nodes(t)
            
            # prediction_mat.append(pred)
            # groundtrue_mat.append(label)
            prediction_mat.append( mysimilarity2GED(pred ,num_nodes_s,num_nodes_t, max_v = max_v))
            groundtrue_mat.append( label)# label
            # print(similarity2GED(label,num_nodes_s,num_nodes_t), train_datasets.ged[s.i,t.i])
            # assert 1 == 0
#             print(label)
#     print(groundtrue_mat)
    #assert 1==0
    prediction_mat = torch.cat(prediction_mat,dim = 0)
    groundtrue_mat = torch.cat(groundtrue_mat,dim = 0)
    p_mat = prediction_mat.reshape(len(datasets_1),-1).T.cpu().numpy()
    gt = groundtrue_mat.reshape(len(datasets_1),-1).T.cpu().numpy() 
    # p_mat = (p_mat*100000).round() / 100000
   #  return train_datasets.ged[:,:900],train_datasets.ged
    return p_mat,gt
 
 


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
   # assert gt.shape[0] == gt.shape[1]
   # Id = gt.shape[0]
   # Id = np.array(list(range(Id)))
   # Id_select = [0]
   # if datasets == "LINUX":
   #    np.random.seed(8)
   #    np.random.shuffle(Id)
   # for i in Id:
   #    if similarity(gt,threshold,Id_select,i):
   #       Id_select.append(i)
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

   train_datasets_all = GEDDataset("../SomeData4kaggle/{}".format(datasetsname), name=datasetsname)
   test_datasets = GEDDataset("../SomeData4kaggle/{}".format(datasetsname), name=datasetsname, train=False)
   max_degree = 0
   for g in train_datasets_all + test_datasets:
      if g.edge_index.size(1) > 0:
         max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
   one_hot_degree = OneHotDegree(max_degree, cat=True)
   train_datasets_all.transform = one_hot_degree
   test_datasets.transform = one_hot_degree
   train_datasets_all = train_datasets_all.shuffle()
   idx = int(len(train_datasets_all)*0.75)
   indices = np.random.permutation(len(train_datasets_all))
   train_datasets,vaild_datasets = train_datasets_all[indices[:idx].tolist()],train_datasets_all[indices[idx:].tolist()]

   # for i in range(20):
   #    print(vaild_datasets[i].i)

   save_file_name = f"./result_norm/{datasets}/{args.model}/sum_none_pGinFalse_woNodeattFalse_woGraphattFalse/minvaild_0.pt"#.format(datasetsname, args.model,args.graph_level_type)
   model = torch.load(save_file_name).to(device)
   with open(f'./result_norm/{datasets}/{args.model}/sum_none_pGinFalse_woNodeattFalse_woGraphattFalse/loss.json', 'r') as f:
      max_v = json.load(f)["max"]
   # if not os.path.exists(f"./landmarkv4/temp/{datasets}_p_mat_train.pt"):
   p_mat_train,gt_train = GetKernel(train_datasets,train_datasets,model,max_v)
   p_mat_val,gt_val = GetKernel(train_datasets,vaild_datasets,model,max_v)
   p_mat_test,gt_test = GetKernel(train_datasets,test_datasets,model,max_v)
   # p_mat_train,gt_train = train_datasets.ged[:900,:900],train_datasets.ged[:900,:900]
   # p_mat_val,gt_val = train_datasets.ged[900:,:900],train_datasets.ged[900:,:900]
   # p_mat_test,gt_test = train_datasets.ged[900:,:900],train_datasets.ged[900:,:900]
   # if not os.path.exists(f"./landmarkv4/temp/"):
   #    os.mkdir(f"./landmarkv4/temp/")
      # torch.save(p_mat_train,f"./landmarkv2/temp/{datasets}_p_mat_train.pt")
      # torch.save(gt_train,f"./landmarkv2/temp/{datasets}_gt_train.pt")
      # torch.save(p_mat_val,f"./landmarkv2/temp/{datasets}_p_mat_val.pt")
      # torch.save(gt_val,f"./landmarkv2/temp/{datasets}_gt_val.pt")
      # torch.save(p_mat_test,f"./landmarkv2/temp/{datasets}_p_mat_test.pt")
      # torch.save(gt_test,f"./landmarkv2/temp/{datasets}_gt_test.pt")

   # p_mat_train = torch.load(f"./landmarkv2/temp/{datasets}_p_mat_train.pt")
   # gt_train = torch.load(f"./landmarkv2/temp/{datasets}_gt_train.pt")
   # p_mat_val = torch.load(f"./landmarkv2/temp/{datasets}_p_mat_val.pt")
   # gt_val = torch.load(f"./landmarkv2/temp/{datasets}_gt_val.pt")
   # p_mat_test = torch.load(f"./landmarkv2/temp/{datasets}_p_mat_test.pt")
   # gt_test = torch.load(f"./landmarkv2/temp/{datasets}_gt_test.pt")
   logger.info("N2AGim GED MSE:")
   logger.info(f" trianing loss of GED generated is {((p_mat_train - gt_train)**2).mean()}, testing loss is {((p_mat_test - gt_test)**2).mean()}")
   
   del model
   torch.cuda.empty_cache()
   # thresholds = {
   #    'AIDS700nef':0.5,#0.43
   #    'LINUX':0.7,#0.7
   #    'IMDBMulti':0.5#0.05
   # }
   # landmark_id = getLandmarkRandom(p_mat_train,args.st_size)
   # if args.st == 'random':
   #    # st_size = 
   landmark_id = getLandmarkRandom(p_mat_train,args.size)
   logger.info(f"length of landmarks is {len(landmark_id)}")
#    CHANNELS_LENGTH = len(landmark_id)
   print(landmark_id)
   # np.save(f"./landmark/save/{datasets}_{args.size}_landmarkid.npy",landmark_id)
   landmark_features_train = p_mat_train[:,landmark_id]

   train_pair = []
   train_target = []
   for ids,i in enumerate(landmark_features_train):
      for idx,j in enumerate(landmark_features_train):
         train_pair.append(process_features(i,j)) 
         train_target.append(gt_train[ids,idx])
   train_pair = torch.from_numpy(np.array(train_pair))
   train_target = torch.from_numpy(np.array(train_target))
   train_tensordatset = TensorDataset(train_pair,train_target)


   landmark_features_val = p_mat_val[:,landmark_id]
   val_pair = []
   val_target = []
   for ids,i in enumerate(landmark_features_val):
      for idx,j in enumerate(landmark_features_train):
         val_pair.append(process_features(i,j))
         val_target.append(gt_val[ids,idx])
   # val_pair = np.array(val_pair)
   # val_df = pd.DataFrame(val_pair, columns = [str(i) for i in range(len(val_pair[0]))])
   # val_df['target'] = val_target
   val_pair = torch.from_numpy(np.array(val_pair))
   val_target = torch.from_numpy(np.array(val_target))
   val_tensordatset = TensorDataset(val_pair,val_target)


   landmark_features_test = p_mat_test[:,landmark_id]
   test_pair = []
   test_target = []
   for ids,i in enumerate(landmark_features_test):
      for idx,j in enumerate(landmark_features_train):
         test_pair.append(process_features(i,j))
         test_target.append(gt_test[ids,idx])
   # test_pair = np.array(test_pair)
   # test_df = pd.DataFrame(test_pair, columns = [str(i) for i in range(len(test_pair[0]))])
   # test_df['target'] = test_target
   test_pair = torch.from_numpy(np.array(test_pair))
   test_target = torch.from_numpy(np.array(test_target))
   test_tensordatset = TensorDataset(test_pair,test_target)


   train_nodes,val_nodes,test_nodes = getnodelist(train_datasets),getnodelist(vaild_datasets),getnodelist(test_datasets)
   node_dict = dict(train = train_nodes,val = val_nodes, test=test_nodes, test_graphs_size = len(test_datasets))
   return train_tensordatset,val_tensordatset,test_tensordatset,node_dict
####################################


class MLP(nn.Module):
   def __init__(self,input_channels,hidden_channels,num_mlp):
      super(MLP,self).__init__()
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
      
   def forward(self,x):
      x = self.mlp_head(x)
      for mlp_block in self.mlp_blocks:
         x = x + mlp_block(x)
      return self.mlp_tail(x)

# def unnorm_(x,max_v = None):
#     if max_v is None:
#        max_v = torch.Tensor([32.2292]).to(x.device)
   
#     return torch.exp(-10**((1 - x)*torch.log10(max_v + 1)) + 1)

def vaild_step(net,loader,node_dict):
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
      pbar = tqdm(loader)
      for idx,(x,y) in enumerate(pbar):
         x = x.to(device)
         y = y.to(device).double().unsqueeze(1)
         y_pred = net(x)
         predition.append(y_pred)
         target.append(y)
      predition = transform2GT(torch.cat(predition,dim = 0),node_lists)
      target = transform2GT(torch.cat(target,dim = 0),node_lists)
      loss_arr = F.mse_loss(predition,target)
   net.train()
   return torch.mean(loss_arr)

def test_step(net,loader,node_dict):
   net.eval()
   loss_arr = []
   node_lists = []
   for i in node_dict.get("test"):
      for j in node_dict.get("train"):
         node_lists.append((i+j)/2)
   node_lists = torch.Tensor(node_lists)
   predition = []
   target = []
   with torch.no_grad():
      pbar = tqdm(loader)
      for idx,(x,y) in enumerate(pbar):
         x = x.to(device)
         y = y.to(device).double().unsqueeze(1)
         y_pred = net(x)
         predition.append(y_pred)
         target.append(y)
      predition = transform2GT(torch.cat(predition,dim = 0),node_lists).numpy().reshape(node_dict["test_graphs_size"],-1)
      target = transform2GT(torch.cat(target,dim = 0),node_lists).numpy().reshape(node_dict["test_graphs_size"],-1)
      rho_list, tau_list, prec_at_10_list, prec_at_20_list = metric(predition, target,verbose = True)
      rho,tau,prec_at_10,prec_at_20 = np.mean(rho_list), np.mean(tau_list), np.mean(prec_at_10_list), np.mean(prec_at_20_list)
      mse = ((predition - target)**2).mean()
   net.train()
   return rho,tau,prec_at_10,prec_at_20,mse

if __name__ == '__main__':
   # data
   logger,work_dir = getLogger(args.dataset)
   logger.info(vars(args))
   device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
   train_tensordatset,val_tensordatset,test_tensordatset,node_dict = data_process()
   # logger.info("load dataset done")
   # model

   net = MLP(input_channels=2*CHANNELS_LENGTH,hidden_channels=128,num_mlp=3).to(device)
   net = net.double()
   optim = torch.optim.Adam(lr=args.lr, params=net.parameters())
   
   train_loader = DataLoader(dataset=train_tensordatset,shuffle=True,batch_size=args.batch)
   val_loader = DataLoader(dataset=val_tensordatset,shuffle=False,batch_size=args.batch)
   test_loader = DataLoader(dataset=test_tensordatset,shuffle=False,batch_size=args.batch)
   
   min_vaild_loss = torch.Tensor([1e10])
   for epoch in range(args.epochs):
      pbar = tqdm(train_loader)
      for idx,(x,y) in enumerate(pbar):
         x = x.to(device)
         y = y.to(device)
         y = y.double().unsqueeze(1)
         y_pred = net(x)
         loss = F.mse_loss(y,y_pred)
         loss.backward()
         optim.step()
         optim.zero_grad()
         pbar.set_description("training loss is {:.6f}".format(loss.item()))
         # logger.info("training loss is {:.6f}".format(loss.item()))
      val_loss = vaild_step(net,val_loader,node_dict)
      if val_loss < min_vaild_loss:
         min_vaild_loss = val_loss
         torch.save(net,f"{work_dir}/{args.dataset}_{args.size}_minvaild.pt")
      # test_loss = vaild_step(net,test_loader,node_dict,'test')
      tqdm.write(f"epoch {epoch}, val loss {val_loss}, min vaild loss {min_vaild_loss}")
      logger.info(f"epoch {epoch}, val loss {val_loss}, min vaild loss {min_vaild_loss}")
   print(f"{work_dir}/{args.dataset}_{args.size}_minvaild.pt")
   net = torch.load(f"{work_dir}/{args.dataset}_{args.size}_minvaild.pt")
   rho,tau,prec_at_10,prec_at_20,mse = test_step(net,test_loader,node_dict)
   logger.info("------------------------------------------------")
   logger.info("mse is {}  10^-3".format(mse*1000))
   logger.info("rho is {}".format(rho))
   logger.info("tau is {}".format(tau))
   logger.info("p@10 is {}".format(prec_at_10))
   logger.info("p@20 is {}".format(prec_at_20))
   
       
 
   # mnist_train, mnist_val = random_split(dataset, [55000, 5000])

   # train_loader = DataLoader(mnist_train, batch_size=32)
   # val_loader = DataLoader(mnist_val, batch_size=32)

   # # model
   # model = LitAutoEncoder()

   # # training
   # trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
   # trainer.fit(model, train_loader, val_loader)
