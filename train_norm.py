from multiprocessing import context
import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.datasets import GEDDataset
from torch_geometric.data import DataLoader
# from torch import nn
import torch.nn.functional as F
# from torch_geometric.nn import SAGEConv,GCNConv
from torch_geometric.transforms import OneHotDegree
# from torch_geometric.data import Batch
# from torchinfo import summary
# from scipy.stats import spearmanr, kendalltau
from torch_geometric.utils import softmax, degree, sort_edge_index,to_dense_batch
import os
from torch.optim import lr_scheduler

import datetime
import prettytable as pt

#my import
from arg_parser import getargs
from GraphSim import graphsim
from EGSCS import EGSCStudent
from EGSCT import EGSCTeacher
from utils import *
from sgim import SGim
from simgnn import simgnn
from n2gim import N2Gim


def train(model,epochs=300):
    
    metric_to_recorder = {"config":vars(args)}
    train_loss = []
    vaild_loss = []
    scheduler = lr_scheduler.StepLR(optim,step_size=70,gamma = 0.8)

    for epoch in range(epochs):
        train_data_all = list(myproduct(train_datasets, train_datasets))
        dl = DataLoader(train_data_all, shuffle=True, batch_size=args.batch_size, num_workers=10,pin_memory=True)
        del train_data_all
        pbar = tqdm(dl) if args.verbose else dl
        model.train()
        for s, t in pbar:
            
            train_epoch_loss = []
            s = s.to(device)
            t = t.to(device)
            if epoch % 2 == 0:
                pred = model(t, s)
            else:
                pred = model(s, t)
            # print(pred)
            
            label = ged[s.i, t.i].to(device)
            
            # print(label)
            # print(pred,label)
            loss = F.mse_loss(pred, label)
            loss.backward()
            optim.step()
            optim.zero_grad()
            train_epoch_loss.append(loss.item())
            if args.verbose:
                pbar.set_description("epoch: {:>3d}/{} ,train loss is  {:.6f}".format(epoch,epochs,loss.item() ))
        if not args.verbose and epoch <= args.vaild_epoch:
            print("epoch: {:>3d}/{} ,train loss is  {:.6f}".format(epoch,epochs,np.mean(train_epoch_loss)))
        metric_to_recorder[f"{epoch}"] = {"training mse":np.mean(train_epoch_loss)}
        if epoch > args.vaild_epoch :
            train_temp_loss = np.mean(train_epoch_loss)
            vaild_epoch_loss = vaild(vaild_datasets,model)
            # test_epoch_loss = vaild_test(test_datasets)
            train_loss.append(train_temp_loss)
            vaild_loss.append(vaild_epoch_loss)
            if vaild_epoch_loss == np.min(vaild_loss):
                torch.save(model, save_file_name + 'minvaild_{}.pt'.format(k))
                torch.save(optim, save_file_name + 'minvaild_optim_{}.pt'.format(k))
            tqdm.write(
                "epoch: {:>3d}/{} train loss: {:.6f} vaild loss: {:.7f} min vaild loss is {:.7f}".format(
                    epoch, epochs, train_temp_loss, vaild_epoch_loss, np.min(vaild_loss)))
            # if np.argmin(vaild_loss) < epoch - args.early_stop:
            #     print("early stopping...")
            #     break
            metric_to_recorder[f"{epoch}"] = {"train mse" : train_temp_loss, "vaild mse" : vaild_epoch_loss}
        dict2json(f'{save_file_name}/loss.json',metric_to_recorder)
        # scheduler.step()
    tqdm.write("testing...")
    torch.save(model, save_file_name + 'final_{}.pt'.format(k))
    torch.save(optim, save_file_name + 'final_optim_{}.pt'.format(k))
    mse,rho,tau,prec_at_10,prec_at_20 = test(model.__class__.__name__)
    metric_to_recorder['test'] = {
        "mse":mse,'rho':rho,"tau":tau,"prec_at_10":prec_at_10,"prec_at_20":prec_at_20
    }
    # metric_to_recorder['test_round'] = {
    #     "mse x 10^-3":mse,'rho':rho,"tau":tau,"prec_at_10":prec_at_10,"prec_at_20":prec_at_20
    # }
    # dict2json(f'{save_file_name}/loss.json',metric_to_recorder)
    end_time = datetime.datetime.now()
    used_time = end_time - start_time
    # metric_to_recorder["used time"] = str(used_time)
    return mse,rho,tau,prec_at_10,prec_at_20,metric_to_recorder

# def vaild(vaild_datasets,model):
#     vaild_data_all = list(myproduct(train_datasets, vaild_datasets))
#     dl = DataLoader(vaild_data_all, shuffle=True, batch_size=args.batch_size, num_workers=10,pin_memory=True)
#     del vaild_data_all
#     loss = []
#     model.eval()
#     with torch.no_grad():
#         pbar = tqdm(dl) if args.verbose else dl
#         for s, t in pbar:
#             s = s.to(device)
#             t = t.to(device)
#             pred = model(s, t)
#             label = ged[s.i, t.i].to(device)
#             lossdata = F.mse_loss(pred, label, reduction='none').cpu().numpy()
#             loss.extend(lossdata)
#             if args.verbose:
#                 pbar.set_description("vaild loss is {:.6f}".format(np.mean(lossdata)))
#     # tqdm.write("mse = {}".format(np.mean(loss) * 0.5))
#     return np.mean(loss)

# def transform2GT(x):
#    return torch.exp(-((1-torch.log(x))*std + mean))
# def transform2GT(x):
#    return torch.exp(10*torch.log(x))

def vaild(vaild_datasets,model):
    # model = torch.load(save_file_name + 'minvaild_{}.pt'.format(k))
    prediction_mat = []
    groundtrue_mat = []

    test_data_all = list(myproduct(train_datasets,vaild_datasets))
    dl = DataLoader(test_data_all,shuffle=False,batch_size=args.batch_size,num_workers = 10,pin_memory = True)
    # del test_data_all
    # print(len(test_data_all))
    loss = []
    model.eval()
    # idx = torch.ones(600,200)
    # idx = []
    with torch.no_grad():
        pbar = tqdm(dl) if args.verbose else dl
        for s,t in pbar:
            s = s.to(device)
            t = t.to(device)
            pred = model(s,t)
            label = ged[s.i,t.i].to(device)
            pred = unnorm_(pred,max_v)
            label = unnorm_(label,max_v)
            # print("unnorm pred :", pred)
            # print("unnorm label:", label)
            # print("exp normged: ", torch.exp(-train_datasets.norm_ged)[s.i,t.i])
            # pred.clamp_(0,1)
            # print(pred.min())
            
            # print(label.max())
            # print(train_datasets.ged[s.i,t.i].max())
            prediction_mat.append(pred)
            groundtrue_mat.append(label)
            # idx[s.i,t.i - 600] = 1
            # idx.append([s.i.cpu().numpy().tolist(),t.i.cpu().numpy().tolist()])
            lossdata = F.mse_loss(pred,label,reduction = 'none').cpu().numpy()
            # loss.extend(lossdata)
            # print(F.mse_loss(pred,label))
            if args.verbose:
                pbar.set_description("vaild loss is {:.6f}".format(np.mean(lossdata)))
        #assert 1 == 0
        prediction_mat = torch.cat(prediction_mat,dim = 0)
        groundtrue_mat = torch.cat(groundtrue_mat,dim = 0)

        p_mat = prediction_mat.reshape(len(train_datasets),len(vaild_datasets)).T.cpu().numpy()
        gt = groundtrue_mat.reshape(len(train_datasets),len(vaild_datasets)).T.cpu().numpy()

        # rho_list, tau_list, prec_at_10_list, prec_at_20_list = metric(p_mat, gt,verbose = args.verbose)
        loss = F.mse_loss(prediction_mat,groundtrue_mat).cpu().numpy()

        tqdm.write("------------------------------------------------")
        tqdm.write("mse is {}  10^-3".format(loss*1000))
        # tqdm.write("rho is {}".format(np.mean(rho_list)))
        # tqdm.write("tau is {}".format(np.mean(tau_list)))
        # tqdm.write("p@10 is {}".format(np.mean(prec_at_10_list)))
        # tqdm.write("p@20 is {}".format(np.mean(prec_at_20_list)))
    model.train()
    return loss



def test(modelname):

    model = torch.load(save_file_name + 'minvaild_{}.pt'.format(k))
    prediction_mat = []
    groundtrue_mat = []
    # print(len(test_datasets))
    test_data_all = list(myproduct(train_datasets,test_datasets))
    dl = DataLoader(test_data_all,shuffle=False,batch_size=args.batch_size,num_workers = 10,pin_memory = True)
    # del test_data_all
    # print(len(test_data_all))
    loss = []
    model.eval()
    # idx = torch.ones(600,200)
    # idx = []
    with torch.no_grad():
        pbar = tqdm(dl) if args.verbose else dl
        for s,t in pbar:
            s = s.to(device)
            t = t.to(device)
            pred = model(s,t)
            label = ged[s.i,t.i].to(device)
            pred = unnorm_(pred,max_v)
            
            label = unnorm_(label,max_v)
            prediction_mat.append(pred)
            groundtrue_mat.append(label)
            # idx[s.i,t.i - 600] = 1
            # idx.append([s.i.cpu().numpy().tolist(),t.i.cpu().numpy().tolist()])
            lossdata = F.mse_loss(pred,label,reduction = 'none').cpu().numpy()
            # loss.extend(lossdata)
            if args.verbose:
                pbar.set_description("test loss is {:.6f}".format(np.mean(lossdata)))
        # idx = np.concatenate(idx,axis = 1)
        # print(np.array(idx[0]).T)
        # print(np.array(idx[1]).T)
        # print(len(idx),len(np.unique(idx,axis = 0)))
        # pd.DataFrame(idx).to_excel("test_idx.xlsx")
        # print(idx)
        # print(np.unique(idx,axis = 0))
        prediction_mat = torch.cat(prediction_mat,dim = 0)
        groundtrue_mat = torch.cat(groundtrue_mat,dim = 0)
        # print(prediction_mat.shape,groundtrue_mat.shape)
        # print(prediction_mat)#.T.
        p_mat = prediction_mat.reshape(len(train_datasets),len(test_datasets)).T.cpu().numpy()
        gt = groundtrue_mat.reshape(len(train_datasets),len(test_datasets)).T.cpu().numpy()
        # print(p_mat)
        rho_list, tau_list, prec_at_10_list, prec_at_20_list = metric(p_mat, gt,verbose = args.verbose)
        loss = F.mse_loss(prediction_mat,groundtrue_mat).cpu().numpy()
        # print(loss)
        tqdm.write("------------------------------------------------")
        tqdm.write("mse is {}  10^-3".format(loss*1000))
        tqdm.write("rho is {}".format(np.mean(rho_list)))
        tqdm.write("tau is {}".format(np.mean(tau_list)))
        tqdm.write("p@10 is {}".format(np.mean(prec_at_10_list)))
        tqdm.write("p@20 is {}".format(np.mean(prec_at_20_list)))
    return loss*1000,np.mean(rho_list),np.mean(tau_list),np.mean(prec_at_10_list),np.mean(prec_at_20_list)


def makehtml(args):
    end_time = datetime.datetime.now()
    used_time = end_time - start_time
    keys = vars(args)
    html_meaasge = ''
    for key,value in keys.items():
        html_meaasge += str(key) + ' : ' + str(value) + '<br/>'
    html_meaasge += "use time = {} <br/>".format(str(used_time)) 
    print("used time is {} ".format(str(used_time)))
    return html_meaasge


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

def norm_(x,max_v = None):
    # x = x + 1
    if max_v is None:
        length =int(x.shape[0]*0.8)
        max_v = x[:,:length].max()

    return 1 - torch.log10(x + 1)/torch.log10(max_v + 1),max_v

def unnorm_(x,max_v):
    return torch.exp(-10**((1 - x)*torch.log10(max_v + 1)) + 1)

if __name__ == "__main__":

    args = getargs()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    start_time = datetime.datetime.now()
    start = time.time()
    start_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(start))

    datasetsname, lr = args.datasets,args.lr

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

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
    modelname = globals()[args.model]
    PrintInfo(train_datasets_all, args)

    mse_arr = []
    rho_arr = []
    tau_arr = []
    prec_at_10_arr = []
    prec_at_20_arr = []
    idx = int(len(train_datasets_all)*0.75)
    indices = np.random.permutation(len(train_datasets_all))
    train_datasets,vaild_datasets = train_datasets_all[indices[:idx].tolist()],train_datasets_all[indices[idx:].tolist()]
    for i in range(10):
        print(vaild_datasets[i].i)
    
    save_file_name = f"./result_norm/{datasetsname}/{args.model}/{args.graph_level_pooling}_{args.node_level_pooling}_pGin{args.pGin}_woNodeatt{args.drop_nb_att}_woGraphatt{args.drop_gb_att}/"#.format(datasetsname, args.model,args.graph_level_type)
    
    try:
        for k in range(args.k_cross):
            
        
            tqdm.write("*****************k_corss : {}****************".format(k))
            # print(modelname)
            model = modelname(train_datasets_all.num_features,args).to(device)
            if not os.path.exists(save_file_name):
                os.makedirs(save_file_name)
            # os.system('cp  *.py   {}'.format(save_file_name))
            # assert 1 == 0
            # for m in model.modules():
            #     if isinstance(m, (nn.Conv2d, nn.Linear)):
            #         nn.init.xavier_uniform_(m.weight)
            optim = torch.optim.Adam(lr=lr, params=model.parameters())
            norm_ged = train_datasets.norm_ged
            # norm_ged_shape = int(norm_ged.shape[0]*0.8)
            # mean,std = norm_ged[:,:norm_ged_shape].mean(),norm_ged[:,:norm_ged_shape].std()
            # norm_ged = (norm_ged - mean) / std + 1
            #ged = torch.exp(-0.1*norm_ged).to(device)
            ged,max_v = norm_(norm_ged,max_v=None)
            # print(max_v)
            # assert 1 == 0
            # print(ged)
            # print(ged[:,:900].max())
            # print(ged[:,:900].min())
            # assert 1 == 0
            mse,rho,tau,prec_at_10,prec_at_20,metric_to_recorder = train(model,args.epochs)
            
            
            end_time = datetime.datetime.now()
            used_time = end_time - start_time
            metric_to_recorder["used time"] = str(used_time)
            metric_to_recorder["max"] = max_v.item()
            #metric_to_recorder['mean'] = mean
            #metric_to_recorder['std'] = std
            dict2json(f'{save_file_name}/loss.json',metric_to_recorder)

            mse_arr.append(mse)
            rho_arr.append(rho)
            tau_arr.append(tau)
            prec_at_10_arr.append(prec_at_10)
            prec_at_20_arr.append(prec_at_20)

            del model
        df = dict(mse = mse_arr,rho = rho_arr, tau = tau_arr,p10 = prec_at_10_arr,p20 = prec_at_20_arr)
        df = pd.DataFrame(df)
        df.to_excel(f"{save_file_name}"+"result.xlsx")
        # mse_result = '{} +- {} <br\>'.format(np.mean(mse_arr),np.std(mse_arr))
        # rho_result = '{} +- {} <br\>'.format(np.mean(rho_arr),np.std(rho_arr))
        # tau_result = '{} +- {} <br\>'.format(np.mean(tau_arr),np.std(tau_arr))
        # prec_at_10_result = '{} +- {} <br\>'.format(np.mean(prec_at_10_arr),np.std(prec_at_10_arr))
        # prec_at_20_result = '{} +- {} <br\>'.format(np.mean(prec_at_20_arr),np.std(prec_at_20_arr))

        # tqdm.write(mse_result.replace('<br/>',''))
        # tqdm.write(rho_result.replace('<br/>',''))
        # tqdm.write(tau_result.replace('<br/>',''))
        # tqdm.write(prec_at_10_result.replace('<br/>',''))
        # tqdm.write(prec_at_20_result.replace('<br/>',''))
    except Exception as e:
        raise e
