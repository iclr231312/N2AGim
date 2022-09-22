import xxlimited
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import  GINConv
from torch_geometric.nn import global_mean_pool,global_add_pool

class EFN(torch.nn.Module):
    def __init__(self, in_channels,out_channels):
        super(EFN, self).__init__()
        reduction = 4
        self.Attention = nn.Sequential(
                        nn.Linear(in_channels,  in_channels // reduction), # downsample
                        nn.ReLU(inplace = True),
                        nn.Linear(in_channels // reduction, in_channels),  #  upsample
                        nn.Tanh()
         )
        self.MLP = nn.Sequential(
             nn.Linear(in_channels,out_channels),
             nn.ReLU(inplace = True)
         )

    def forward(self,*h):
        # assert h_i
        # print(h)
        h_ij = torch.cat(list(h),dim=1)
        encoding = self.Attention(h_ij)*h_ij + h_ij
        joint_embeddings = self.MLP(encoding)

        return joint_embeddings

class global_attention_readout(nn.Module):
    def __init__(self,in_channels) -> None:
        super(global_attention_readout,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels,in_channels),
            nn.Sigmoid()
        )
    def forward(self,graph_i_representation,batch):

        c_i = global_mean_pool(graph_i_representation,batch)
        c_i = self.fc(c_i)
        c_i = c_i[batch]
        h_i = global_add_pool( torch.mul(torch.sigmoid(torch.sum(graph_i_representation * c_i,dim = 1)).unsqueeze(1),graph_i_representation),batch)
        
        return h_i


class EGSCStudent(torch.nn.Module):
    def __init__(self, in_channels,args):
        super(EGSCStudent, self).__init__()
        D = 16
        self.args = {
                "out_channels" : 1,
                "filters_1": 64,
                "filters_2": 32,
                "filters_3": 16,
                "d":16,
                "distill":True
        }
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, self.args['filters_1']), 
            torch.nn.ReLU(), 
            torch.nn.Linear(self.args['filters_1'], self.args['filters_1']),
            torch.nn.BatchNorm1d(self.args['filters_1'])
            )
        
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(self.args['filters_1'], self.args['filters_2']), 
            torch.nn.ReLU(), 
            torch.nn.Linear(self.args['filters_2'], self.args['filters_2']),
            torch.nn.BatchNorm1d(self.args['filters_2'])
            )
        
        nn3 = torch.nn.Sequential(
            torch.nn.Linear(self.args['filters_2'], self.args['filters_3']), 
            torch.nn.ReLU(), 
            torch.nn.Linear(self.args['filters_3'], self.args['filters_3']),
            torch.nn.BatchNorm1d(self.args['filters_3'])
            )
        
        self.conv_1 = GINConv(nn1, train_eps=True)
        self.conv_2 = GINConv(nn2, train_eps=True)
        self.conv_3 = GINConv(nn3, train_eps=True)

        # global attention
        self.global_attention = global_attention_readout(self.args['filters_3']) 
        # EFN
        self.EFNs = EFN(self.args['filters_3']*2,D)

        self.fc = nn.Sequential(
            nn.Linear(D,D//2),
            nn.ReLU(inplace=True),
            nn.Linear(D//2,1)
            # nn.Sigmoid()
        )

    def joint_embeddings(self,x_i,x_j,edge_index_i,edge_index_j,x_i_batch,x_j_batch):
        xi,xj = x_i,x_j
        
        xi1 = self.conv_1(xi, edge_index_i)
        xj1 = self.conv_1(xj, edge_index_j)

        xi2 = self.conv_2(xi1, edge_index_i)
        xj2 = self.conv_2(xj1, edge_index_j)

        xi3 = self.conv_3(xi2, edge_index_i)
        xj3 = self.conv_3(xj2, edge_index_j)
        joint_embeddings3 = self.EFNs(self.global_attention(xi3,x_i_batch),self.global_attention(xj3,x_j_batch))

        h_ij_all = self.EFNs[3](joint_embeddings3)
        return h_ij_all

    def decomposition(self,data1,data2):
        h_AB_all = self.joint_embeddings( data1.x,data2.x, data1.edge_index,data2.edge_index,data1.batch,data2.batch)
        h_AA_all = self.joint_embeddings( data1.x,data1.x, data1.edge_index,data1.edge_index,data1.batch,data1.batch)
        h_BB_all = self.joint_embeddings( data2.x,data2.x, data2.edge_index,data2.edge_index,data2.batch,data2.batch)

        h_Ab = h_AB_all - h_BB_all
        h_aB = h_AB_all - h_AA_all
        return h_Ab,h_aB

    def forward(self,data1,data2):
        xi = data1.x
        xj = data2.x
        edge_index_i = data1.edge_index
        edge_index_j = data2.edge_index
        batch_i = data1.batch
        batch_j =data2.batch
        
        xi1 = self.conv_1(xi, edge_index_i)
        xj1 = self.conv_1(xj, edge_index_j)

        xi2 = self.conv_2(xi1, edge_index_i)
        xj2 = self.conv_2(xj1, edge_index_j)

        xi3 = self.conv_3(xi2, edge_index_i)
        xj3 = self.conv_3(xj2, edge_index_j)
        h_i = self.global_attention(xi3,batch_i)
        h_j = self.global_attention(xj3,batch_j)
        h_AB = self.EFNs(h_i,h_j)

        if self.args['distill']:
            h_AA = self.EFNs(h_i,h_i)
            h_BB = self.EFNs(h_j,h_j)
            h_Ab = h_AB - h_BB
            h_aB = h_AB - h_AA
            return self.fc( h_AB ).squeeze(1),h_Ab,h_aB
        else:
            return self.fc( h_AB ).squeeze(1)
    def GetEmbeddings(self,data1):
        xi = data1.x
        edge_index_i = data1.edge_index
        batch_i = data1.batch
        xi1 = self.conv_1(xi, edge_index_i)
        xi2 = self.conv_2(xi1, edge_index_i)
        xi3 = self.conv_3(xi2, edge_index_i)
        h_i = self.global_attention(xi3,batch_i)
        
        return h_i

    def GetFcResult(self,h):
        h_AB = self.EFNs(h)
        return self.fc( h_AB ).squeeze(1)

    def __repr__(self) -> str:
        return f"EGSCS"