import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

"""
修改过 process_feature"""
def parse_index_file(filename):
    """
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """
    Create mask.
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    
    # x.shape:(140, 1433); y.shape:(140, 7);tx.shape:(1000, 1433);ty.shape:(1708, 1433);
    # allx.shape:(1708, 1433);ally.shape:(1708, 7)
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder) #从小到大排序

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        # test_idx_range_full=1015
        '''
        A=np.array([[1,0,2,0],[0,0,0,0],[3,0,0,0],[1,0,0,4]])
        AS=sp.lil_matrix(A)
        print(AS)
        # (0, 0) 1
        # (0, 2) 2
        # (2, 0) 3
        # (3, 0) 1
        # (3, 3) 4
        '''
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        
        #test_idx_range-min(test_idx_range):列表中每个元素都减去min(test_idx_range)，即将test_idx_range列表中的index值变为从0开始编号
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        # tx.shape : (1015,3703),其中997,994,993,980,938...等15行全为0
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    # 将allx和tx叠起来并转化成LIL格式的feature,即输入一张整图
    features = sp.vstack((allx, tx)).tolil()
    # 把特征矩阵还原，和对应的邻接矩阵对应起来，因为之前是打乱的，不对齐的话，特征就和对应的节点搞错了。    
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # 邻接矩阵格式也是LIL的，并且shape为(2708, 2708)    
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # len(list(idx_val)) + len(list(idx_train)) + len(idx_test) =  1640
    idx_test = test_idx_range.tolist()  #array_to_list:range(1708,2707)
    idx_train = range(len(y))   #range(0,140)
    idx_val = range(len(y), len(y)+500)  # range(140,640)

    # 训练mask：idx_train=[0,140)范围的是True，后面的是False
    train_mask = sample_mask(idx_train, labels.shape[0])
    # print(train_mask,train_mask.shape)
    # [True  True  True... False False False]    

    # 验证mask：val_mask的idx_val=(140, 640]范围为True，其余的是False    
    val_mask = sample_mask(idx_val, labels.shape[0])
    # test_mask，idx_test=[1708,2707]范围是True，其余的是False    
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape) #(2708,7)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    # 替换了true位置    
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


# 返回一个格式为(coords, values, shape)的元组
def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()  #将此矩阵转换为坐标格式。
        # print(np.vstack((features.row, features.col)))
        # array([[   0,    0,    0, ..., 2707, 2707, 2707],[1274, 1247, 1194, ...,  329,  186,   19]], dtype=int32)
        # print(np.vstack((features.row, features.col)).transpose())
        # array([[0, 1274],[0, 1247],[0, 1194],...,[2707,329],[2707,186],[2707,19]], dtype=int32)
        coords = np.vstack((mx.row, mx.col)).transpose()  #coords:坐标
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):    #>>>a = 2,>>> isinstance (a,int),返回True
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

# 特征矩阵的每一行的每个元素除以行和，处理后的每一行元素之和为1
# 处理特征矩阵，跟谱图卷积的理论有关，目的是要把周围节点的特征和自身节点的特征都捕捉到，同时避免不同节点间度的不均衡带来的问题
def preprocess_features(features):
    """
    Row-normalize feature matrix and convert to tuple representation
    """
    # a.sum()是将矩阵中所有的元素进行求和;a.sum(axis = 0)是每一列列相加;a.sum(axis = 1)是每一行相加    
    rowsum = np.array(features.sum(1)) # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1).flatten() # 1/rowsum, [2708]. flatten()函数返回一个折叠成一维的数组。但是该函数只能适用于numpy对象，即array或者mat，普通的list列表是不行的。
    # r_inv: [0.11111111 0.04347826 0.05263158... 0.05555556 0.07142857 0.07692308]    
    
    r_inv[np.isinf(r_inv)] = 0. # zero inf data(inf表示无穷大，即rowsum=0的情况)   np.isinf(ndarray)返回一个判断是否是无穷的bool型数组
    r_mat_inv = sp.diags(r_inv) # sparse diagonal matrix, [2708, 2708]
    features = r_mat_inv.dot(features) # D^-1:[2708, 2708]@features:[2708, 1433] , np.dot表示矩阵乘积
    features = sp.coo_matrix(features)
    return sparse_to_tuple(features) # [coordinates, data, shape], []


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj) #将adj转化为坐标格式和权重。
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    #torch.eye()返回一个2维张量，对角线位置全1，其它位置全0，将output对角线全为0
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)





def chebyshev_polynomials(adj, k):
    """
    Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation).
    """
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM') #计算最大特征值，_为特征向量
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])  

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
