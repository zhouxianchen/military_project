import torch
from torch import nn
from torch.nn import functional as F
from layer import GraphConvolution


class GCN(nn.Module):

    #以cora为例，input_dim=feat_dim=1433，output_dim=num_classes=7，num_features_nonzero=49216
    def __init__(self, input_dim, output_dim, num_features_nonzero,args):
        super(GCN, self).__init__()

        self.input_dim = input_dim # 1433
        self.output_dim = output_dim # 7

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        print('num_features_nonzero:', num_features_nonzero)

        # args.hidden=16, args.dropout=0.5, output_dim=7
        self.layers = nn.Sequential(GraphConvolution(self.input_dim, args.hidden, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=True,
                                                     bias=False),

                                    GraphConvolution(args.hidden, output_dim, num_features_nonzero,
                                                     activation=lambda x: x,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=False,
                                                     bias=False),

                                    )

    def forward(self, inputs):
        x, support = inputs
        x = x.float()
        x = self.layers((x, support))

        return x

    def l2_loss(self):
# 让参数简单，防止过拟合
        layer = self.layers.children()
        #print("layers_children:",layer)
        layer = next(iter(layer))
        #print("next(iter(layer)):",layer)
        loss = None
        
        # p就是w1(1433,16)和w2(16,7)
        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss
