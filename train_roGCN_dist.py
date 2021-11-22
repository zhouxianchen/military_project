from DeepRobust.deeprobust.graph.defense.gcn import GCN
from DeepRobust.deeprobust.graph.utils import preprocess
from MyRo_GAT import Pre_GAT
from RoGCN import ProGCN
import argparse
import os
import handle_data
import dgl
import scipy.sparse as sp
import torch.nn.functional as F
import networkx as nx
import torch
import pickle
from utils import save_result
from handle_data import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help='The random seed')
parser.add_argument('--dist','-dist', type=float, default=50000, help='The dtw threshold')
parser.add_argument('--sim_time', default=3600,type=float, help="The dtw_computing_time")
parser.add_argument('--model', default='gcn')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--ptb_rate', type=float, default=0.25)
parser.add_argument('--weight_decay', type=float, default=5e-4) #（权重衰减）：目的就是为了让权重减少到更小的值，在一定程度上减少模型过拟合的问题
parser.add_argument('--early_stopping', type=int, default=15)
parser.add_argument('--max_degree', type=int, default=3) # K阶的切比雪夫近似矩阵的参数k
parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
parser.add_argument('--only_gcn', action='store_true',
        default=False, help='test the performance of gcn without other components')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
parser.add_argument('--lambda_', type=float, default=1, help='weight of feature smoothing')
parser.add_argument('--id_feature', default=True,help="choosee feature")
parser.add_argument('--noise_level', type=float, default=0,help="noise leavel")
args = parser.parse_args()


def generate_dtw_with_true_label(args, qb_information,unit_information,noise_level=args.noise_level):
    np.random.seed(args.seed)
    handle = Handle_data(qb_information,sim_time=args.sim_time,noise_level=noise_level)
    save_name = os.path.join("save_networkx", str(qb_information[-8:-4] + "_" + str(args.dist) + "_" + str(noise_level) + "_"  + "_" + str(args.sim_time) +str(args.seed)+ ".pb"))
    if os.path.exists(save_name): ###避免每次加载，但是修改了就要注意名称。
        with open(save_name, "rb") as f:
            nx_g = pickle.load(f)
        current_type = handle.generate_current_type()
    else:
        nx_g = handle.generate_G(dist=args.dist)
        with open(save_name, "wb") as f:
            pickle.dump(nx_g,f)
        current_type = handle.current_type
    Red = handle_data.Handle_test_label(unit_information)
    nx_g = Red.obtain_test_label(nx_g, current_type=current_type)
    handle.draw(nx_g,is_dtw=False)
    return nx_g

qb_information = "./logdata2/logdata/blue_qb_202011242026.txt"
unit_information = "./logdata2/logdata/red_units_202011242026.txt"
# qb_information = r"C:\Users\zhouxianchen\PycharmProjects\deeplearning\logdata2\taskB\blue_qb_202012011041.txt"
# unit_information = r"C:\Users\zhouxianchen\PycharmProjects\deeplearning\logdata2\taskB\red_units_202012011041.txt"
G = generate_dtw_with_true_label(args, qb_information, unit_information)

adj, features,labels,idx_train,idx_test,feature_id = data_for_rogcn_training(G)

device = 'cpu'
###添加了噪声
# from deeprobust.graph.global_attack import Random
# attacker = Random()
# n_perturbations = int(args.ptb_rate * (adj.sum() // 2))
# adj = attacker.attack(adj, n_perturbations, type='add')
# args.only_gcn=True
if args.only_gcn:
    feature2 = features
    features = feature_id
    feature_id = feature2
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout, device=device)
    labels = torch.LongTensor(labels)
    adj = torch.FloatTensor(adj.todense())
    rogcn = ProGCN(model, args, device)
    rogcn.fit(features, adj, labels, idx_train, idx_test, feature_id)
    acc = rogcn.test(features, labels, idx_test)
else:
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout, device=device)
    labels = torch.LongTensor(labels)
    adj = torch.FloatTensor(adj.todense())
    rogcn = ProGCN(model,args,device)
    rogcn.fit(features,adj,labels,idx_train,idx_test,feature_id)
    acc = rogcn.test(features, labels, idx_test)

if args.only_gcn:
    save_result(acc,"./save_acc/save_result_gcn_dist.txt")
else:
    save_result(acc,"./save_acc/save_result_rogcn_dist.txt")