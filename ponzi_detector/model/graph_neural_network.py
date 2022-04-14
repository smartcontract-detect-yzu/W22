import torch
from torch_geometric.nn import CGConv, GlobalAttention, GATConv, AGNNConv
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList

torch.manual_seed(8)


class CGCClass(torch.nn.Module):
    def __init__(self, model_params):
        super(CGCClass, self).__init__()
        feature_size = model_params["MODEL_FEAT_SIZE"]
        self.n_layers = model_params["MODEL_LAYERS"]
        self.dropout_rate = model_params["MODEL_DROPOUT_RATE"]
        dense_neurons = model_params["MODEL_DENSE_NEURONS"]
        edge_dim = model_params["MODEL_EDGE_DIM"]
        out_channels = model_params["MODEL_OUT_CHANNELS"]

        self.gnn_layers = ModuleList([])

        # CGC block ??
        # self.cgc1 = CGConv(feature_size)

        # CGC, Transform, BatchNorm block
        for i in range(self.n_layers):
            self.gnn_layers.append(
                CGConv(feature_size, dim=edge_dim, batch_norm=True)
            )

        # Linear layers
        self.linear1 = Linear(feature_size, dense_neurons)  # 100 48
        self.bn2 = BatchNorm1d(dense_neurons)
        self.linear2 = Linear(dense_neurons, out_channels)  # 48 48

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Initial CGC ??
        # x = self.cgc1(x, edge_index, edge_attr)

        for i in range(self.n_layers):
            x = self.gnn_layers[i](x, edge_index, edge_attr)

        # Pooling
        x = global_max_pool(x, batch)

        # Output block
        x = F.dropout(x, p=0.0, training=self.training)  # dropout_rate
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
