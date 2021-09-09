import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from networks import graph
from matplotlib import pyplot as plt


# import pdb

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1./math.sqrt(self.weight(1))
        # self.weight.data.uniform_(-stdv,stdv)
        torch.nn.init.xavier_uniform_(self.weight)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv,stdv)

    def forward(self, input, adj=None, relu=False):
        support = torch.matmul(input, self.weight)
        # print(support.size(),adj.size())
        if adj is not None:
            output = torch.matmul(adj, support)
        else:
            output = support
        # print(output.size())
        if self.bias is not None:
            return output + self.bias
        else:
            if relu:
                return F.relu(output)
            else:
                return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Featuremaps_to_Graph(nn.Module):

    def __init__(self, input_channels, hidden_layers, nodes=7):
        super(Featuremaps_to_Graph, self).__init__()
        self.pre_fea = Parameter(torch.FloatTensor(input_channels, nodes))
        self.weight = Parameter(torch.FloatTensor(input_channels, hidden_layers))
        self.reset_parameters()

    def forward(self, input):
        assert input.dim() in [4, 5]
        if input.dim() == 4:
            n, c, h, w = input.size()
            # print('fea input',input.size())
            input1 = input.view(n, c, h * w)
            input1 = input1.transpose(1, 2)  # n x hw x c
            # print('fea input1', input1.size())
            ############## Feature maps to node ################
            fea_node = torch.matmul(input1, self.pre_fea)  # n x hw x n_classes

            # written by Shumao Pang
            fea_logit = fea_node.transpose(1, 2)  # n x n_classes x hw
            fea_logit = fea_logit.view(n, fea_node.size()[2], h, w)  # n x n_classes x h x w

            weight_node = torch.matmul(input1, self.weight)  # n x hw x hidden_layer
            # softmax fea_node
            fea_node = F.softmax(fea_node, dim=-1) # n x hw x n_classes

            # Witten by Shumao Pang
            fea_sum = torch.sum(fea_node, dim=1).unsqueeze(1).expand(n, h * w, self.pre_fea.size()[-1])
            fea_node = torch.div(fea_node, fea_sum)

            # print(fea_node.size(),weight_node.size())
            graph_node = F.relu(torch.matmul(fea_node.transpose(1, 2), weight_node))
            return graph_node, fea_logit  # n x n_class x hidden_layer
        elif input.dim() == 5:
            n, c, d, h, w = input.size()
            # print('fea input',input.size())
            input1 = input.view(n, c, d * h * w)
            input1 = input1.transpose(1, 2)  # n x dhw x c
            # print('fea input1', input1.size())
            ############## Feature maps to node ################
            fea_node = torch.matmul(input1, self.pre_fea)  # n x dhw x n_classes

            # written by Shumao Pang
            fea_logit = fea_node.transpose(1, 2)  # n x n_classes x dhw
            fea_logit = fea_logit.view(n, fea_node.size()[2], d, h, w)  # n x n_classes x d x h x w

            weight_node = torch.matmul(input1, self.weight)  # n x dhw x hidden_layer
            # softmax fea_node
            fea_node = F.softmax(fea_node, dim=-1) # n x dhw x n_classes

            # Witten by Shumao Pang
            fea_sum = torch.sum(fea_node, dim=1).unsqueeze(1).expand(n, d * h * w, self.pre_fea.size()[-1])
            fea_node = torch.div(fea_node, fea_sum)

            # print(fea_node.size(),weight_node.size())
            graph_node = F.relu(torch.matmul(fea_node.transpose(1, 2), weight_node))  # n x n_class x hidden_layer
            return graph_node, fea_logit  # n x n_class x hidden_layer, n x n_classes x d x h x w

    def reset_parameters(self):
        for ww in self.parameters():
            torch.nn.init.xavier_uniform_(ww)


class Graph_to_Featuremaps(nn.Module):
    def __init__(self, hidden_layers, output_channels, dimension=2):
        super(Graph_to_Featuremaps, self).__init__()
        self.hidden_layers = hidden_layers
        self.output_channels = output_channels
        if dimension == 3:
            self.conv = nn.Conv3d(hidden_layers, self.output_channels, 1, bias=False)
            self.bn = nn.BatchNorm3d(output_channels)
        else:
            self.conv = nn.Conv2d(hidden_layers, self.output_channels, 1, bias=False)
            self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(True)

    def forward(self, graph, fea_logit):
        '''
        :param graph: batch x nodes x hidden_layers
        :param fea_logit: batch x nodes x h x w
        :return: fea_map: batch x output_channels x h x w
        '''
        assert fea_logit.dim() in [4, 5]
        if fea_logit.dim() == 4:
            fea_prob = F.softmax(fea_logit, dim=1)  # batch x nodes x h x w
            batch, nodes, h, w = fea_prob.size()
            fea_prob = fea_prob.view(batch, nodes, h * w)  # batch x nodes x hw
            fea_prob = fea_prob.transpose(1, 2)  # batch x hw x nodes
            fea_map = torch.matmul(fea_prob, graph)  # batch x hw x hidden_layer
            fea_map = fea_map.transpose(1, 2)  # batch x hidden_layers x hw
            fea_map = fea_map.view(batch, self.hidden_layers, h, w)  # batch x hidden_layers x h x w
            fea_map = self.conv(fea_map)  # batch x output_channels x h x w
            fea_map = self.bn(fea_map)
            fea_map = self.relu(fea_map)
            return fea_map
        else:
            fea_prob = F.softmax(fea_logit, dim=1)  # batch x nodes x d x h x w
            batch, nodes, d, h, w = fea_prob.size()
            fea_prob = fea_prob.view(batch, nodes, d * h * w)  # batch x nodes x dhw
            fea_prob = fea_prob.transpose(1, 2)  # batch x dhw x nodes
            fea_map = torch.matmul(fea_prob, graph)  # batch x dhw x hidden_layer
            fea_map = fea_map.transpose(1, 2)  # batch x hidden_layers x dhw
            fea_map = fea_map.view(batch, self.hidden_layers, d, h, w)  # batch x hidden_layers x d x h x w
            fea_map = self.conv(fea_map)  # batch x output_channels x d x h x w
            fea_map = self.bn(fea_map)
            fea_map = self.relu(fea_map)
            return fea_map


if __name__ == '__main__':
    graph = torch.randn((7, 128))
    pred = (torch.rand((7, 7)) * 7).int()
    # a = en.forward(graph,pred)
    # print(a.size())
