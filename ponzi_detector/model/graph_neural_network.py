import torch
from torch_geometric.nn import GraphConv, GlobalAttention, GATConv, AGNNConv, GatedGraphConv, RGCNConv
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList
from imblearn.over_sampling import SMOTE

torch.manual_seed(8)

from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Linear

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch.nn import Parameter
import math
from typing import Any


def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)


class GAL(MessagePassing):
    def __init__(self, in_features, out_featrues):

        # 进行加权求和
        super(GAL, self).__init__(aggr='add')

        # 定义attention参数a
        self.a = torch.nn.Parameter(torch.zeros(size=(2 * out_featrues, 1)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = torch.nn.LeakyReLU()
        self.linear = torch.nn.Linear(in_features, out_featrues)

    def forward(self, x, edge_index):

        # 特征映射
        x = self.linear(x)
        N = x.size()[0]
        col, row = edge_index

        # 将相邻接点的特征拼接，然后计算e值
        a_input = torch.cat([x[row], x[col]], dim=1)

        # print('a_input.size', a_input.size())

        # 将规模压缩到一维
        temp = torch.mm(a_input, self.a).squeeze()

        # print('temp.size', temp.size())

        e = self.leakyrelu(temp)

        # print('e', e)
        # print('e.size', e.size())

        # e_all为同一个节点与其全部邻居的计算的分数的和，用于计算归一化softmax
        e_all = torch.zeros(x.size()[0])
        count = 0
        for i in col:
            e_all[i] += e[count]
            count = count + 1

        # print('e_all', e_all)

        # 计算alpha值
        for i in range(len(e)):
            e[i] = math.exp(e[i]) / math.exp(e_all[col[i]])

        # print('attention', e)
        # print('attention.size', e.size())

        # 传递信息
        return self.propagate(edge_index, x=x, norm=e)

    def message(self, x_j, norm):

        # print('x_j:', x_j)
        # print('x_j.size', x_j.size())
        # print('norm', norm)
        # print('norm.size', norm.size())
        # print('norm.view.size', norm.view(-1, 1).size())

        # 计算求和项
        return norm.view(-1, 1) * x_j


class MYGNN(MessagePassing):
    def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 0,
                 aggr: str = 'add', batch_norm: bool = False,
                 bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.channels = channels
        self.dim = dim  #
        self.batch_norm = batch_norm
        self.leakyrelu = torch.nn.LeakyReLU()

        # if isinstance(channels, int):
        #     channels = (channels, channels)

        # The learnable parameters to compute attention coefficients:
        # self.att = Parameter(torch.Tensor(1, 1, channels))

        # self.attention = GAL(channels, channels)
        # self.dim_edge = 16

        # self.line_edge_in = Linear(self.dim, 8)
        # self.line_edge_hidden = Linear(16, 8)
        # self.line_edge_out = Linear(8, self.dim)

        self.line_attention = Linear(channels + self.dim, channels, bias=bias)
        self.line_node = Linear(channels + self.dim, channels, bias=bias)
        if batch_norm:
            self.bn = BatchNorm1d(channels)
        else:
            self.bn = None
        self.active = torch.nn.ReLU(inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.att_src)
        self.line_attention.reset_parameters()
        self.line_node.reset_parameters()
        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # edge_attr = self.line_edge_in(edge_attr)
        # edge_attr = self.line_edge_hidden(edge_attr)
        # edge_attr = self.line_edge_out(edge_attr)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out if self.bn is None else self.bn(out)
        out += x[1]

        # out = self.attention(out, edge_index)
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        if edge_attr is None:
            z = torch.cat([x_i], dim=-1)
        else:
            z = torch.cat([x_i, edge_attr], dim=-1)

        return self.line_attention(z).sigmoid() * F.softplus(self.line_node(z))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels}, dim={self.dim})'


class CGCClass(torch.nn.Module):
    def __init__(self, model_params):
        super(CGCClass, self).__init__()
        feature_size = model_params["MODEL_FEAT_SIZE"]
        self.n_layers = model_params["MODEL_LAYERS"]
        self.dropout_rate = model_params["MODEL_DROPOUT_RATE"]
        dense_edge_neurons = model_params["MODEL_EDGE_DENSE_NEURONS"]
        edge_neurons = model_params["MODEL_EDGE_NEURONS"]
        dense_neurons = model_params["MODEL_DENSE_NEURONS"]
        edge_dim = model_params["MODEL_EDGE_DIM"]
        out_channels = model_params["MODEL_OUT_CHANNELS"]
        self.training = True
        self.gnn_layers = ModuleList([])

        if edge_dim != 0:
            self.linear_edge1 = Linear(edge_dim, dense_edge_neurons)  # 4 --> 8
            self.linear_edge2 = Linear(dense_edge_neurons, edge_neurons)  # 8 --> 8
            self.edge_flag = 1
            self.edge_dim = edge_neurons
        else:
            self.edge_flag = 0
            self.edge_dim = 0

        # # 0. GGNN
        # self.ggnn_layer = GatedGraphConv(feature_size, 3)
        # self.model_name = "GGNN"

        # # 1. GATConv
        # for i in range(self.n_layers):
        #     self.gnn_layers.append(
        #         GATConv(feature_size, feature_size)
        #     )
        # self.model_name = "GATConv"

        # # 2. GCN
        # for i in range(self.n_layers):
        #     self.gnn_layers.append(
        #         # MYGNN(feature_size, dim=self.edge_dim, batch_norm=True)
        #         GraphConv(feature_size, feature_size)
        #     )
        # self.model_name = "GCN"

        # 3. RS-GCN
        for i in range(self.n_layers):
            self.gnn_layers.append(
                MYGNN(feature_size, dim=self.edge_dim, batch_norm=True)
            )
        self.model_name = "RS-GCN"

        # Linear layers
        self.linear1 = Linear(feature_size, dense_neurons)  # 100 48
        self.bn2 = BatchNorm1d(dense_neurons)
        self.smote = SMOTE(random_state=42)
        self.linear2 = Linear(dense_neurons, out_channels)  # 48 2

        print("#### current model is {} #####".format(self.model_name))
        print("\n")
        print("\n")

    def forward(self, data):

        x, y, edge_index, edge_attr, batch = data.x, data.y, data.edge_index, data.edge_attr, data.batch

        # Initial CGC ??
        # x = self.cgc1(x, edge_index, edge_attr)

        if self.edge_flag == 1:
            edge_attr = self.linear_edge1(edge_attr)
            edge_attr = self.linear_edge2(edge_attr)
            # pass  # 在图卷积层进行
        else:
            edge_attr = None

        # # GGNN
        # self.ggnn_layer(x, edge_index)

        # # GATConv
        # for i in range(self.n_layers):
        #     x = self.gnn_layers[i](x, edge_index)

        # # GCN
        # for i in range(self.n_layers):
        #     x = self.gnn_layers[i](x, edge_index)

        # RS-GCN
        for i in range(self.n_layers):
            x = self.gnn_layers[i](x, edge_index, edge_attr)

        # Pooling
        x = global_max_pool(x, batch)

        # Output block
        x = F.dropout(x, p=self.dropout_rate, training=self.training)  # dropout_rate
        x = torch.relu(self.linear1(x))
        x = self.bn2(x)
        x = self.linear2(x)

        if torch.isnan(torch.mean(self.linear2.weight)):
            raise RuntimeError("Exploding gradients. Tune learning rate")

        x = torch.sigmoid(x)  # 二分类，输出约束在(0, 1)

        return x


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    correct = 0
    loss = 0.
    criterion = torch.nn.CrossEntropyLoss()
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        label = data.y.argmax(dim=1)
        # print("pred {}, label {}".format(pred, label))
        # print("out {}, data.y {}".format(out, data.y))
        batch_loss = criterion(out, data.y)
        correct += int((pred == label).sum())
        loss += batch_loss
    return correct / len(loader.dataset), loss / len(loader.dataset)


dataset_info = {
    "cfg": {
        "root": 'examples/ponzi_src/dataset/cfg'
    },
    "sliced": {
        "root": 'examples/ponzi_src/dataset/sliced'
    },
    "etherscan": {
        "root": 'examples/ponzi_src/dataset/etherscan'
    }
}
# if __name__ == '__main__':
#
#     train_type = "sliced"  # "cfg"
#     test_type = "cfg"  # "sliced"
#
#     root_dir = dataset_info[train_type]["root"]
#     train_valid_dataset = PonziDataSet(root=root_dir, dataset_type=train_type)
#
#     root_dir = dataset_info[test_type]["root"]
#     test_dataset = PonziDataSet(root=root_dir, dataset_type=test_type)
#     test_off_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
#
#     feature_size = train_valid_dataset[0].x.shape[1]
#
#     model_params = {
#         "MODEL_FEAT_SIZE": feature_size,
#         "MODEL_LAYERS": 3,
#         "MODEL_DROPOUT_RATE": 0.1,
#         "MODEL_DENSE_NEURONS": 48,
#         "MODEL_EDGE_DIM": 0,
#         "MODEL_OUT_CHANNELS": 2  # 每一类的概率
#     }
#
#     solver = {
#         "SOLVER_LEARNING_RATE": 0.001,
#         "SOLVER_SGD_MOMENTUM": 0.8,
#         "SOLVER_WEIGHT_DECAY": 0.001
#     }
#
#     train_size = int(len(train_valid_dataset) * 0.7)
#     valid_size = len(train_valid_dataset) - train_size
#     print("train_size:{} valid_size:{}".format(train_size, valid_size))
#     train_dataset, valid_dataset = torch.utils.data.random_split(train_valid_dataset, [train_size, valid_size])
#
#     train_off_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = CGCClass(model_params=model_params)
#     model = model.to(device)
#
#     optimizer = torch.optim.Adam(model.parameters(),
#                                  lr=solver["SOLVER_LEARNING_RATE"],
#                                  weight_decay=solver["SOLVER_WEIGHT_DECAY"])
#     criterion = torch.nn.CrossEntropyLoss()
#
#     print("训练集：{}   测试集：{}".format(train_type, test_type))
#
#     epochs = 16
#     for epoch in range(epochs):
#         model.train()
#         training_loss = 0
#         for i, data in enumerate(train_off_loader):
#             optimizer.zero_grad()
#             data = data.to(device)
#             out = model(data)
#             target = data.y
#             loss = criterion(out, target)
#             training_loss += loss.item() * data.num_graphs
#             loss.backward()
#             optimizer.step()
#             # print("epoch {} batch {} {} Training loss: {}".format(epoch, i, data.num_graphs, loss.item()))
#         training_loss /= len(train_off_loader.dataset)
#         print("epoch {} Training loss: {}".format(epoch, training_loss))
#
#     valid_off_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
#     val_acc, val_loss = test(model, valid_off_loader, device)
#     print("normal Validation loss: {}\taccuracy:{}".format(val_loss, val_acc))
#
#     test_off_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
#     val_acc, val_loss = test(model, test_off_loader, device)
#     print("\n\nTEST loss: {}\taccuracy:{}".format(val_loss, val_acc))
#
#     dataset_type = "etherscan"
#     root_dir = dataset_info[dataset_type]["root"]
#     test_dataset = PonziDataSet(root=root_dir, dataset_type=dataset_type)
#     test_off_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
#     val_acc, val_loss = test(model, test_off_loader, device)
#     print("etherscan test loss: {}\taccuracy:{}".format(val_loss, val_acc))
