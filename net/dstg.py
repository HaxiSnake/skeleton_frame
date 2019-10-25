# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.autograd import Variable
from collections import OrderedDict

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, kernel_size, **kwargs):
        super(_DenseLayer, self).__init__()
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        self.add_module('norm1', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('st_gcn', st_gcn(bn_size * growth_rate, growth_rate, kernel_size, 1, **kwargs)) 
    def forward(self, prev_features, A):
        bottleneck_output = self.relu1(self.norm1(self.conv1(prev_features)))
        new_features, _ = self.st_gcn(bottleneck_output, A)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=(2,1), stride=(2,1)))
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, 
                growth_rate, kernel_size, A, edge_importance_weighting=True, edge_importance_depth=1, **kwargs):
        super(_DenseBlock, self).__init__()
        # self.A = A
        if edge_importance_weighting:
            assert num_layers%edge_importance_depth == 0
            self.edge_importance_depth = edge_importance_depth
        else:
            self.edge_importance_depth = 1
        self.register_buffer('A', A)
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                kernel_size=kernel_size,
                **kwargs
            )
            self.add_module('denselayer%d' % (i + 1), layer)
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in range(num_layers//self.edge_importance_depth)
            ])
        else:
            self.edge_importance = [1] * len(num_layers//self.edge_importance_depth) 

    def forward(self, init_features):
        features = [init_features]
        for idx, child in enumerate(self.named_children()):
            name, layer = child
            if "denselayer" not in name:
                break
            concated_features = torch.cat(features, 1)
            importance = self.edge_importance[idx//self.edge_importance_depth]
            new_features = layer(concated_features, A=self.A * importance)
            features.append(new_features)
        return torch.cat(features, 1)
        # for child, importance in zip(self.named_children(),self.edge_importance):
        #     name, layer = child
        #     concated_features = torch.cat(features, 1)
        #     new_features = layer(concated_features, A=self.A * importance)   
        #     features.append(new_features)
        # return torch.cat(features, 1)


class DSTG(nn.Module):
    r"""Dense_ST_GCN model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    and st_gcn
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        num_class (int) - number of classification classes
    """
    def __init__(self, graph_args, growth_rate=12, depth=148, compression=0.5,
                 num_init_features=24, bn_size=4,
                 num_class=10, in_channels=2, tsize=256, **kwargs ):

        super(DSTG, self).__init__()
        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        # self.register_buffer('A', A)
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        if (depth - 4) % 3:
            raise Exception('Invalid depth')
        block_config = [(depth - 4) // 6 for _ in range(3)]
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = (tsize//8,A.size(1))

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=(9,1), stride=(2,1), padding=(4,0), bias=False)),
        ]))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                **kwargs,
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                kernel_size=kernel_size,
                A=A,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_class)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(N, M, -1, 1, 1).mean(dim=1)
        out = out.squeeze()
        # out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.classifier(out)
        return out

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


