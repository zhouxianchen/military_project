import torch
from torch import nn
from torch.nn import functional as F



# 其中 mask 是一个索引向量，值为1表示该位置的标签在训练数据中是给定的；比如100个数据中训练集已知带标签的数据有50个，
# 那么计算损失的时候，loss 乘以的 mask  等于 loss 在未带标签的地方都乘以0没有了，而在带标签的地方损失变成了mask倍；
# 即只对带标签的样本计算损失。
# 注：loss的shape与mask的shape相同，等于样本的数量：(None,），所以 loss *= mask 是向量点乘。

def masked_loss(out, label, mask):

    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()  
    mask = mask / mask.mean()  # mask.mean()= mask.sum()/adj.shape[0]，扩大了mask.mean()倍，因此要除以这个数
    loss *= mask
    loss = loss.mean()
    return loss


def masked_acc(out, label, mask):
    # [node, f]
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()  # 比较pred和label是否相等，相等为1，不相等为0
    mask = mask.float()
    mask = mask / mask.mean()  
    correct *= mask
    acc = correct.mean()
    return acc



def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate: dropout=0.5
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device) # 0.5+随机数
    dropout_mask = torch.floor(random_tensor).byte()  #torch.floor:向下取整
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    # [2, 49216] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask] #留下dropout_mask中为1的信息
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
#    tensor(indices=tensor([[   0,    0,    1,  ..., 2707, 2707, 2707],
#                           [1194,   81, 1177,  ...,  754,  447,  186]]),
#           values=tensor([0.1111, 0.1111, 0.0435,  ..., 0.0769, 0.0769, 0.0769]),
#           device='cuda:0', size=(2708, 1433), nnz=24675, layout=torch.sparse_coo)
    
    out = out * (1./ (1-rate)) 
#    tensor(indices=tensor([[   0,    0,    1,  ..., 2707, 2707, 2707],
#                           [1194,   81, 1177,  ...,  754,  447,  186]]),
#           values=tensor([0.2222, 0.2222, 0.0870,  ..., 0.1538, 0.1538, 0.1538]),
#           device='cuda:0', size=(2708, 1433), nnz=24675, layout=torch.sparse_coo)    

    return out


def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)  #torch.mm()矩阵相乘

    return res


def save_result(result,name,*args):
    with open(name,"a+") as f:
        for arg in args:
            f.write(arg)
            f.write('')
        f.write(str(result))
        f.write('\n')