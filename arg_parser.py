import argparse
from xmlrpc.client import boolean
def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default='AIDS700nef')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument("--num_gnn",type=int,default=3)
    # parser.add_argument("--cnn_reshape", type=int, default=30)
    parser.add_argument("--verbose",action='store_true')
    parser.add_argument("--model",type = str ,default='SGSim')
    parser.add_argument("--k_cross",type = int,default=1)
    parser.add_argument("--vaild_epoch",type = int,default = 150)
    # parser.add_argument("--alpha",type=float, default = 0)
    parser.add_argument("--seed",type = int,default = 2)
    parser.add_argument("--doc",type = str)
    # parser.add_argument("--name",type=str,default='full')
    
    
    parser.add_argument("--pGin",action='store_true')
    parser.add_argument("--drop_nb_att",action='store_true')
    parser.add_argument("--drop_gb_att",action='store_true')
    parser.add_argument("--graph_level_pooling",type=str,default='none',help='sum/max/mean/att/none')
    parser.add_argument("--node_level_pooling",type=str,default='none',help='so/cp/none')
    args = parser.parse_args()
    return args