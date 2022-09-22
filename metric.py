import time
import numpy as np
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
import pandas as pd
import prettytable as pt
import inspect
#my import
from arg_parser import getargs
from GraphSim_model import GraphSim
from SFGNNGraphSim_model import SFGNNGraphSim
from SFGraphSim_model import SFGraphSim
from utils import *
from SFCGNNGraphSim_model import SFCGNNGraphSim

def train(model,epochs=300):
    train_loss = []
    vaild_loss = []
    for epoch in range(epochs):
        train_data_all = list(myproduct(train_datasets, train_datasets))
        dl = DataLoader(train_data_all, shuffle=True, batch_size=args.batch_size, num_workers=10,pin_memory=True)
        del train_data_all
        pbar = tqdm(dl) if args.verbose else dl
        for s, t in pbar:
            train_epoch_loss = []
            s = s.to(device)
            t = t.to(device)
            if epoch % 2 == 0:
                pred = model(t, s)
            else:
                pred = model(s, t)
            label = ged[s.i, t.i]
            loss = F.mse_loss(pred, label)
            loss.backward()
            optim.step()
            optim.zero_grad()
            train_epoch_loss.append(loss.item())
            # if args.verbose:
            #     pbar.set_description("train loss is  {:.6f}".format(loss.item() * 0.5))

        train_temp_loss = np.mean(train_epoch_loss) * 0.5
        vaild_epoch_loss = vaild()
        train_loss.append(train_temp_loss)
        vaild_loss.append(vaild_epoch_loss)
        if vaild_epoch_loss == np.min(vaild_loss):
            torch.save(model, './{}_{}_{}/minvaild_{}.pt'.format(datasetsname, model.__class__.__name__, start,k))
            torch.save(optim, './{}_{}_{}/minvaild_optim_{}.pt'.format(datasetsname, model.__class__.__name__, start,k))
        tqdm.write(
            "epoch: {:>3d}/{} train loss: {:.6f} test loss: {:.7f} min test loss is {:.7f}".format(
                epoch, epochs, train_temp_loss, vaild_epoch_loss, np.min(vaild_loss)))
        if np.argmin(vaild_loss) < epoch - args.early_stop:
            print("early stopping...")
            break

    tqdm.write("testing...")
    torch.save(model, './{}_{}_{}/final_{}.pt'.format(datasetsname, model.__class__.__name__, start,k))
    torch.save(optim, './{}_{}_{}/final_optim_{}.pt'.format(datasetsname, model.__class__.__name__, start,k))
    mse,rho,tau,prec_at_10,prec_at_20 = test(model.__class__.__name__)
    return mse,rho,tau,prec_at_10,prec_at_20

def vaild():
    test_data_all = list(myproduct(train_datasets, test_datasets))
    dl = DataLoader(test_data_all, shuffle=True, batch_size=args.batch_size, num_workers=10,pin_memory=True)
    del test_data_all
    loss = []
    with torch.no_grad():
        pbar = tqdm(dl) if args.verbose else dl
        for s, t in pbar:
            s = s.to(device)
            t = t.to(device)
            pred = model(s, t)
            label = ged[s.i, t.i].to(device)
            lossdata = F.mse_loss(pred, label, reduction='none').cpu().numpy()
            loss.extend(lossdata)
            if args.verbose:
                pbar.set_description("test loss is {:.6f}".format(np.mean(lossdata) * 0.5))
    # tqdm.write("mse = {}".format(np.mean(loss) * 0.5))
    return np.mean(loss) * 0.5


def test(modelname):
    model = torch.load('./IMDBMulti_SFCGNNGraphSim_1617718136.7832766/minvaild_0.1.pt').to(device)
    prediction_mat = np.zeros(ged.shape)
    train_data_all = list(myproduct(train_datasets,test_datasets))
    dl = DataLoader(train_data_all,shuffle=True,batch_size=args.batch_size,num_workers = 10,pin_memory = True)
    del train_data_all
    loss = []
    with torch.no_grad():
        pbar = tqdm(dl) if args.verbose else dl
        for s,t in pbar:
            s = s.to(device)
            t = t.to(device)
            pred = model(s,t)
            label = ged[s.i,t.i].to(device)
            prediction_mat[s.i.cpu(),t.i.cpu()] = pred.cpu()
            prediction_mat[t.i.cpu(),s.i.cpu()] = pred.cpu()
            lossdata = F.mse_loss(pred,label,reduction = 'none').cpu().numpy()
            loss.extend(lossdata)
            if args.verbose:
                pbar.set_description("test loss is {:.6f}".format(np.mean(lossdata)*0.5))
        p_mat = prediction_mat[len(train_datasets):, :len(train_datasets)]
        gt = ged[len(train_datasets):, :len(train_datasets)].cpu().numpy()
        rho_list, tau_list, prec_at_10_list, prec_at_20_list = metric(p_mat, gt,verbose = args.verbose)
        tqdm.write("------------------------------------------------")
        tqdm.write("mse is {} 10^-3".format(np.mean(loss)*0.5*1000))
        tqdm.write("rho is {}".format(np.mean(rho_list)))
        tqdm.write("tau is {}".format(np.mean(tau_list)))
        tqdm.write("p@10 is {}".format(np.mean(prec_at_10_list)))
        tqdm.write("p@20 is {}".format(np.mean(prec_at_20_list)))
    return np.mean(loss)*0.5*1000,np.mean(rho_list),np.mean(tau_list),np.mean(prec_at_10_list),np.mean(prec_at_20_list)

def makehtml(args):
    keys = vars(args)
    html_meaasge = ''
    for key,value in keys.items():
        html_meaasge += str(key) + ' : ' + str(value) + '<br/>'
    return html_meaasge

def showtable(alphas,mse_arr,rho_arr,tau_arr,prec_at_10_arr,prec_at_20_arr):
    tb =  pt.PrettyTable()
    tb.field_names = ['alpha','mse','rho','tau','p@10','p@20']
    for i in range(len(alphas)):
        tb.add_row([alphas[i],mse_arr[i],rho_arr[i],tau_arr[i],prec_at_10_arr[i],prec_at_20_arr[i]])
    tb.align = 'l'
    print(tb)


def PrintInfo(dataset,args):
    print(f'==============  Dataset INFO: {dataset}  ===========================\n')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    data = dataset[0]  # Get the first graph object.
    print(data)
    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print('\n===================  Args INFO  =================================\n')
    tb =  pt.PrettyTable()

    tb.field_names = ['Hyperparameter','values']
    keys = vars(args)
    for key,value in keys.items():
        tb.add_row([key,value])
    tb.align = 'l'
    print(tb)



if __name__ == "__main__":
    args = getargs()
    start = time.time()
    GNNModel = globals()[args.gnn]
    datasetsname, lr = args.datasets,args.lr

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)
    train_datasets = GEDDataset("../SomeData4kaggle/{}".format(datasetsname), name=datasetsname)
    test_datasets = GEDDataset("../SomeData4kaggle/{}".format(datasetsname), name=datasetsname, train=False)
    max_degree = 0
    for g in train_datasets + test_datasets:
        if g.edge_index.size(1) > 0:
            max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
    one_hot_degree = OneHotDegree(max_degree, cat=True)
    train_datasets.transform = one_hot_degree
    test_datasets.transform = one_hot_degree
    train_datasets = train_datasets.shuffle()
    modelname = globals()[args.model]
    ged = torch.exp(-train_datasets.norm_ged).to(device)
    PrintInfo(train_datasets, args)
    test(modelname)
    # mse_arr = []
    # rho_arr = []
    # tau_arr = []
    # prec_at_10_arr = []
    # prec_at_20_arr = []
    # alphas = [0,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # for alpha in alphas:
    #     k = alpha
    #     tqdm.write("*****************alphas : {}****************".format(alpha))
    #     if 'alpha' in  inspect.getfullargspec(modelname).args:
    #         model = modelname(hidden_channels=args.hidden_size,label_size=1,cnn_channel=3,in_channels = train_datasets.num_features,GNNModel = GNNModel,alpha = alpha).to(device)
    #     else:
    #         model = modelname(hidden_channels=args.hidden_size,label_size=1,cnn_channel=3,in_channels = train_datasets.num_features,GNNModel = GNNModel).to(device)
    #     if not os.path.exists('./{}_{}_{}/'.format(datasetsname, model.__class__.__name__, start)):
    #         os.makedirs('./{}_{}_{}/'.format(datasetsname, model.__class__.__name__, start))
    #     for m in model.modules():
    #         if isinstance(m, (nn.Conv2d, nn.Linear)):
    #             nn.init.xavier_uniform_(m.weight)
    #     optim = torch.optim.Adam(lr=lr, params=model.parameters())
    #     ged = torch.exp(-train_datasets.norm_ged).to(device)
    #     mse,rho,tau,prec_at_10,prec_at_20 = train(model,args.epochs)


    #     mse_arr.append(mse)
    #     rho_arr.append(rho)
    #     tau_arr.append(tau)
    #     prec_at_10_arr.append(prec_at_10)
    #     prec_at_20_arr.append(prec_at_20)

    #     del model
    # showtable(alphas,mse_arr,rho_arr,tau_arr,prec_at_10_arr,prec_at_20_arr)

    # df = pd.DataFrame()
    # df['alpha'] = alphas
    # df['mse'] = mse_arr
    # df['rho'] = rho_arr
    # df['tau'] = tau_arr
    # df['p@10'] = prec_at_10_arr
    # df['p@20'] = prec_at_20_arr
    # df.to_csv('./{}_{}_{}/result.csv'.format(datasetsname, args.model, start))
    # mse_result = '{} +- {} <br\>'.format(np.mean(mse_arr),np.std(mse_arr))
    # rho_result = '{} +- {} <br\>'.format(np.mean(rho_arr),np.std(rho_arr))
    # tau_result = '{} +- {} <br\>'.format(np.mean(tau_arr),np.std(tau_arr))
    # prec_at_10_result = '{} +- {} <br\>'.format(np.mean(prec_at_10_arr),np.std(prec_at_10_arr))
    # prec_at_20_result = '{} +- {} <br\>'.format(np.mean(prec_at_20_arr),np.std(prec_at_20_arr))


    # message = makehtml(
    #     args) + '<br/>' + f"{modelname} result : <br/> mse : {mse_result} <br/> rho : {rho_result} <br/> tau : {tau_result} <br/> prec_at_10 : {prec_at_10_result} <br/> prec_at_20 : {prec_at_20_result}"
    # tqdm.write(mse_result.replace('<br/>',''))
    # tqdm.write(rho_result.replace('<br/>',''))
    # tqdm.write(tau_result.replace('<br/>',''))
    # tqdm.write(prec_at_10_result.replace('<br/>',''))
    # tqdm.write(prec_at_20_result.replace('<br/>',''))
