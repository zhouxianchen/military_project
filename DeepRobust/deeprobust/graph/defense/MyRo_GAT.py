"""
This model is using the GAT with revised A,X，（my method).
"""

import torch
import torch.nn as nn
import scipy.sparse as sp
import torchvision
import numpy as np
import time
import torch.optim as optim
import torch.nn.functional as F
from DeepRobust.deeprobust.graph.utils import accuracy
from DeepRobust.deeprobust.graph.defense.pgd import PGD, prox_operators
from copy import deepcopy
import dgl
from dgl import DGLGraph
import networkx as nx
print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# print(torch.version.cuda)
# print(torch.backends.cudnn.version())
# print(torch.cuda.get_device_name(0))
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
#
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def sparse_diag(input_tensor):
    """
    Returns the identity matrix as a sparse matrix
    """
    size = input_tensor.shape[0]
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size).cuda()
    return torch.sparse.FloatTensor(indices, input_tensor, torch.Size([size, size]))

class Pre_GAT:
    def __init__(self, model, args, device):

        '''
        Compute structure and adjaceny iteratively.
        model: The backbone GNN model in ProGNN

        For Pre_GCN, args.gamma=0 args.lambda=0
        '''
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.best_feature = None
        self.weights = None
        self.estimator = None
        self.estimator2 = None
        self.model = model.to(device)

    def fit(self, features, adj, labels, idx_train, idx_val):
        args = self.args
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        estimator = EstimateAdj(adj, symmetric=args.symmetric).to(self.device)
        estimator2 = EstimateFeature(features).to(self.device)
        self.estimator2 = estimator2
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(estimator.parameters(),
                                       momentum=0.9, lr=args.lr_adj)

        self.optimizer_feat = optim.SGD(estimator2.parameters(),
                                        momentum=0.9, lr=args.lr_adj)
        self.optimizer_l1 = PGD(estimator.parameters(),
                                proxs=[prox_operators.prox_l1],
                                lr=args.lr_adj, alphas=[args.alpha])

        if args.dataset == "pubmed":
            self.optimizer_nuclear = PGD(estimator.parameters(),
                                         proxs=[prox_operators.prox_nuclear_cuda],
                                         lr=args.lr_adj, alphas=[args.beta])
        else:
            self.optimizer_nuclear = PGD(estimator.parameters(),
                                         proxs=[prox_operators.prox_nuclear_cuda],
                                         lr=args.lr_adj, alphas=[args.beta])

        t_total = time.time()
        adj_list = []
        adj_list.append(estimator.estimated_adj.clone().detach().numpy())
        for epoch in range(args.epochs):
            if args.only_gcn:
                self.train_gcn(epoch, features, estimator.estimated_adj, labels, idx_train, idx_val)
            else:
                for i in range(int(args.outer_steps)):
                    self.train_adj(epoch, features, adj, labels, idx_train, idx_val)
                    adj_list.append(estimator.estimated_adj.clone().detach().numpy())
                    torch.cuda.empty_cache()
                for i in range(int(args.outer_steps)):
                    self.train_feat(epoch, features, adj, labels, idx_train, idx_val)
                    torch.cuda.empty_cache()
                for i in range(int(args.inner_steps)):
                    self.train_gcn(epoch, features, adj,
                                   labels, idx_train, idx_val)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(args)

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)
        return adj_list

    def train_adj(self, epoch, features, adj, labels, idx_train, idx_val):
        """

        :param epoch:
        :param features:
        :param adj:
        :param labels:
        :param idx_train:
        :param idx_val:
        :return:
        """
        estimator = self.estimator
        estimator2 = self.estimator2
        args = self.args
        if args.debug:
            print("\n=== train_adj ===")
        t = time.time()
        estimator.train()
        self.optimizer_adj.zero_grad()
        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = torch.norm(estimator.estimated_adj - adj, p='fro')
        g = self.generate_g(estimator.estimated_adj)
        if args.lambda_:
            loss_smooth_feat = self.feature_smoothing(estimator.estimated_adj, features)
        else:
            loss_smooth_feat = 0 * loss_l1
        print(features.dtype)
        torch.cuda.empty_cache()
        output = self.model(g, features)
        loss_fcn = torch.nn.CrossEntropyLoss()
        loss_gcn = loss_fcn(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        #loss_symmetric = torch.norm(estimator.estimated_adj \
                           #         - estimator.estimated_adj.t(), p="fro")
        loss_symmetric = 0 * loss_l1
        loss_diffiential = loss_fro + args.gamma * loss_gcn + args.lambda_ * loss_smooth_feat + args.phi * loss_symmetric
        loss_diffiential.backward()
        self.optimizer_adj.step()
        loss_nuclear = 0 * loss_fro
        if args.beta != 0:
            self.optimizer_nuclear.zero_grad()
            self.optimizer_nuclear.step()
            loss_nuclear = prox_operators.nuclear_norm
        if args.alpha !=0:
            self.optimizer_l1.zero_grad()
            self.optimizer_l1.step()

       # total_loss = loss_fro \
       #              + args.gamma * loss_gcn \
       #              + args.alpha * loss_l1 \
       #              + args.beta * loss_nuclear \
       #              + args.phi * loss_symmetric
       #
       #  estimator.estimated_adj.data.copy_(torch.clamp(
       #      estimator.estimated_adj.data, min=0, max=1))

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        # normalized_adj = estimator.normalize()
        # estimator.estimated_adj.data.copy_(normalized_adj)
        g = self.generate_g(estimator.estimated_adj)
        with torch.no_grad():
            output = self.model(g, features)
        loss_val = loss_fcn(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = g
            self.best_feature = estimator2.estimated_feature.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = g
            self.best_feature = estimator2.estimated_feature.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_fro: {:.4f}'.format(loss_fro.item()),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_feat: {:.4f}'.format(loss_smooth_feat.item()),
                      'loss_symmetric: {:.4f}'.format(loss_symmetric.item()),
                      'delta_l1_norm: {:.4f}'.format(torch.norm(estimator.estimated_adj - adj, 1).item()),
                      'loss_l1: {:.4f}'.format(loss_l1.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()),
                      'loss_nuclear: {:.4f}'.format(loss_nuclear.item()))

    def train_feat(self, epoch, features, adj, labels, idx_train, idx_val):
        estimator2 = self.estimator2
        args = self.args
        if args.debug:
            print("\n === This the train_feature===")
        t = time.time()
        estimator2.train()
        self.optimizer_feat.zero_grad()
        loss_fro = torch.norm(estimator2.estimated_feature - features, p='fro')
        g = self.generate_g(adj)
        output = self.model(g, estimator2.estimated_feature)
        loss_fcn = torch.nn.CrossEntropyLoss()
        loss_gcn = loss_fcn(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        if args.method == "smooth":
             loss_feat = self.feature_smoothing(adj, estimator2.estimated_feature)
        # elif args.methos == "filter":
        #     loss_feat = self.feature_filter(adj, estimator2.estimated_feature)

        loss_diffiential = loss_fro + args.gamma * loss_gcn + args.lambda_ * loss_feat
        # loss_diffiential = loss_fro + args.gamma * loss_gcn
        loss_diffiential.backward()
        self.optimizer_feat.step()

        total_loss = loss_fro \
                 + args.gamma * loss_gcn \
                  + args.lambda_ * loss_feat
        # estimator2.estimated_feature.data.copy(estimator2.estimated_feature.data)

        self.model.eval()
        with torch.no_grad():
            output = self.model(g, estimator2.estimated_feature)

            loss_val = loss_fcn(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
        del output
        torch.cuda.empty_cache()
        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = g
            self.best_feature = estimator2.estimated_feature.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = g
            self.best_feature = estimator2.estimated_feature.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_fro: {:.4f}'.format(loss_fro.item()),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_feat: {:.4f}'.format(loss_feat.item()),
                      'loss_total: {:.4f}'.format(total_loss.item())
                      )

    def train_gcn(self, epoch, features, adj, labels, idx_train, idx_val):
        args = self.args
        estimator = self.estimator
        # adj = estimator.normalize()
        estimator2 = self.estimator2
        features = estimator2.estimated_feature
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        g = self.generate_g(estimator.estimated_adj)
        output = self.model(g, features)
        loss_fcn = torch.nn.CrossEntropyLoss()
        loss_train = loss_fcn(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        with torch.no_grad():
            output = self.model(g, features)

            loss_val = loss_fcn(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = g
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = g
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())
        del g, output
        torch.cuda.empty_cache()
        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))
        
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.4f}'.format(loss_train.item()), 'acc_train: {:.4f}'.format(acc_train.item()), 'loss_val: {:.4f}'.format(loss_val.item()), 'acc_val: {:.4f}'.format(acc_val.item()),  'time: {:.4f}s'.format(time.time() - t))
        # wandb.log({'epoch': epoch+1, 'loss_train': loss_train.item(), 'acc_train': acc_train.item(), 'loss_val': loss_val.item(), 'acc_val': acc_val.item(), 'time': time.time()-t})

    def test(self, features, labels, idx_test):
        print("\t=== testing ===")
        self.model.eval()
        g = self.best_graph
        features = self.best_feature
        args = self.args
     #   if args.dataset == "polblogs":
      #      features = torch.FloatTensor(features)
        if self.best_graph is None:
             #adj = self.estimator.normalize()
            adj = self.estimator.estimated_adj
            estimator2 = self.estimator2
            features = estimator2.estimated_feature
            g = self.generate_g(adj)
        # g = DGLGraph(nx.from_numpy_matrix(adj.detach().numpy(), create_using=nx.DiGraph()))
        # netg = nx.from_numpy_matrix(g.adjacency_matrix().to_dense().numpy(), create_using=nx.MultiGraph)
        # g = DGLGraph()
        # g.from_networkx(netg, edge_attrs=['weight'])
        #
        with torch.no_grad():
            output = self.model(g, features)
            loss_fcn = torch.nn.CrossEntropyLoss()
            loss_test = loss_fcn(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
        print("\tTest set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        # wandb.log({'Test_accuracy': acc_test.item(), 'Test_loss': loss_test.item()})
 
    def feature_filter(self, adj, X):
        """
        Compute the loss of filter,the sum of rank 0 of \|U^TX\|_0
        """
        return X
    
    def feature_smoothing(self, adj, X):
        args = self.args

        adj = (adj.t() + adj) / 2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj
        # if args.dataset == "pubmed":
        #     return torch.trace(torch.matmul(torch.matmul(X.t(), L), X))
        r_inv = r_inv + 1e-3
        r_inv = r_inv.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L
        L2 = r_mat_inv @ L @ r_mat_inv
        del L2
        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)

        return loss_smooth_feat
    
    # def feature_smoothing(self, adj, X):
    #     args = self.args
    #     adj = (adj.t() + adj) / 2
    #     rowsum = adj.sum(1)
    #     r_inv = rowsum.flatten()
    #     D = sparse_diag(r_inv)
    #     L = D - adj
    #     del D, adj
    #    # if args.dataset == "pubmed":
    #    #      return torch.trace(torch.matmul(torch.matmul(X.t(), L), X))
    #     print(r_inv.shape)
    #     r_inv = r_inv + 1e-3
    #     r_inv = r_inv.pow(-1 / 2).flatten()
    #     r_inv[torch.isinf(r_inv)] = 0.
    #     r_mat_inv = torch.diag(r_inv)
    #     del r_inv
    #     torch.cuda.empty_cache()
    #     # L = r_mat_inv @ L
    #     L = torch.sparse.mm(torch.sparse.mm(L, r_mat_inv).t(),r_mat_inv.t()).t()
    #     # L = r_mat_inv @ L @ r_mat_inv
    #     XLXT = torch.sparse.mm(torch.sparse.mm(L,X).t(),X).t()
    #     # XLXT = torch.matmul(torch.matmul(X.t(), L), X)
    #     loss_smooth_feat = torch.trace(XLXT.to_dense())
    #     return loss_smooth_feat

    def generate_g(self, estimated_adj):
        args = self.args
        if args.symmetric:
            adj = (estimated_adj+estimated_adj.t())/2
        else:
            adj = estimated_adj
        del estimated_adj

        a = (adj + torch.eye(adj.shape[0])).detach().cpu().numpy()
        # a = np.eye(adj.shape[0])+adj.to_dense().detach().numpy()
        b = sp.coo_matrix(a)
        del a
        g = DGLGraph(b).to(self.device)
        g.edata['weight'] = b.data
        del b
        torch.cuda.empty_cache()
        return g


class EstimateAdj(nn.Module):

    def __init__(self, adj, symmetric=False):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):

        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t())
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).cuda())
       # normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]))
        # normalized_adj = adj + torch.eye(adj.shape[0])
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx



class EstimateFeature(nn.Module):

    def __init__(self, feature):
        super(EstimateFeature, self).__init__()
        n, k = feature.size()
        self.estimated_feature = nn.Parameter(torch.FloatTensor(n, k))
        self._init_estimation(feature)

    def _init_estimation(self, feature):
        with torch.no_grad():
            n, k = feature.size()
            self.estimated_feature.data.copy_(feature)

    def forward(self):
        return self.estimated_feature

    # def normalize(self):
    #
    #     if self.symmetric:
    #         adj = (self.estimated_adj + self.estimated_adj.t())
    #     else:
    #         adj = self.estimated_adj
    #
    #     # normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).cuda())
    #     normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]))
    #     return normalized_adj

    # def _normalize(self, mx):
    #     rowsum = mx.sum(1)
    #     r_inv = rowsum.pow(-1/2).flatten()
    #     r_inv[torch.isinf(r_inv)] = 0.
    #     r_mat_inv = torch.diag(r_inv)
    #     mx = r_mat_inv @ mx
    #     mx = mx @ r_mat_inv
    #     return mx

