import torch
import torch.nn as nn
import torch.utils.data as Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def calculate_metrics(preds, labels, ponzi_label=0):
    TP = FP = TN = FN = 0

    for batch_preds, batch_labels in zip(preds, labels):
        for pred, label in zip(batch_preds, batch_labels):

            if label == ponzi_label:
                if pred == label:
                    TP += 1
                else:
                    FP += 1
            else:
                if pred == label:
                    TN += 1
                else:
                    FN += 1

    total_data_num = TP + TN + FP + FN

    # 计算acc
    acc = (TP + TN) / (TP + TN + FP + FN)

    # 计算recall
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    # 计算precision
    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    # 计算f1
    if (precision + recall) != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    print("统计值:{} {} {} {}".format(TP, TN, FP, FN))
    return acc, recall, precision, f1, total_data_num


def get_data_from_csv(file_name):
    ponzi_dataset = loadtxt(file_name, delimiter=",")

    X = ponzi_dataset[:, 1:]

    Y_t = ponzi_dataset[:, 0]
    Y = []

    for y in Y_t:
        if y == 1:
            Y_t.append([1, 0])  # [ponzi, no_ponzi]
        else:
            Y_t.append([0, 1])  # [ponzi, no_ponzi]

    return torch.from_numpy(X).float(), torch.from_numpy(np.array(Y)).float()


def change_label_to_binary_classifier(Y_t):
    Y = []

    for y in Y_t:
        if y == 1:
            Y.append([1, 0])  # [ponzi, no_ponzi]
        else:
            Y.append([0, 1])  # [ponzi, no_ponzi]

    return Y


def get_etherscan_data():
    ponzi_dataset_path = "xgboost_dataset_etherscan_ponzi.csv"
    ponzi_dataset = loadtxt(ponzi_dataset_path, delimiter=",")

    X_etherscan = ponzi_dataset[:, 1:]
    Y_etherscan = change_label_to_binary_classifier(ponzi_dataset[:, 0])
    print(X_etherscan)
    return torch.from_numpy(X_etherscan).float(), torch.from_numpy(np.array(Y_etherscan)).float()


def get_ponzi_train_data():
    seed = 7
    test_size = 0.3

    # ponzi_dataset
    ponzi_dataset_path = "xgboost_dataset_all_ponzi.csv"
    ponzi_dataset = loadtxt(ponzi_dataset_path, delimiter=",")

    X_p = ponzi_dataset[:, 1:]
    Y_p = change_label_to_binary_classifier(ponzi_dataset[:, 0])  # [ponzi, no_ponzi]
    X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(X_p, Y_p, test_size=test_size, random_state=seed,
                                                                shuffle=True)

    # no_ponzi_dataset
    no_ponzi_dataset_path = "xgboost_dataset_all_no_ponzi.csv"
    no_ponzi_dataset = loadtxt(no_ponzi_dataset_path, delimiter=",")

    X_np = no_ponzi_dataset[:, 1:]
    Y_np = change_label_to_binary_classifier(no_ponzi_dataset[:, 0])
    X_np_train, X_np_test, y_np_train, y_np_test = train_test_split(X_np, Y_np, test_size=test_size, random_state=seed,
                                                                    shuffle=True)

    # 构建数据集
    X_train = np.concatenate((X_p_train, X_np_train))
    y_train = np.concatenate((y_p_train, y_np_train))

    X_test = np.concatenate((X_p_test, X_np_test))
    y_test = np.concatenate((y_p_test, y_np_test))

    return torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float(), \
           torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()


class LSTM(nn.Module):
    def __init__(self, input_size=78, hidden_layer_size=128, output_size=2, layers=8):
        """
        LSTM二分类任务
        :param input_size: 输入数据的维度
        :param hidden_layer_size:隐层的数目
        :param output_size: 输出的个数
        """
        super().__init__()
        # hidden_layer_size = [128, 64]
        self.input_size = input_size
        self.num_layers = layers
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_layer_size,
                            num_layers=self.num_layers,
                            bias=True,
                            batch_first=True)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x):
        input_x = input_x.view(len(input_x), 1, -1)

        # (direction * num_layers, batch_size, layer_size)
        h_0 = torch.zeros(1 * self.num_layers, input_x.size(0), self.hidden_layer_size)
        c_0 = torch.zeros(1 * self.num_layers, input_x.size(0), self.hidden_layer_size)

        lstm_out, (hn, cn) = self.lstm(input_x, (h_0, c_0))
        predictions = self.linear(lstm_out.view(len(input_x), -1))  # =self.linear(lstm_out[:, -1, :])
        # predictions = self.sigmoid(linear_out)

        return predictions


if __name__ == '__main__':

    # 得到数据
    x_train, y_train, x_valid, y_valid = get_ponzi_train_data()
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x_train, y_train),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=64,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多进程（multiprocess）来读数据
    )

    # 参数：
    epochs = 128
    input_size = 78
    hidden_layer_size = 48
    output_size = 2
    layers = 32
    lr = 0.01

    # 模型
    model = LSTM(input_size=input_size, hidden_layer_size=hidden_layer_size, output_size=output_size, layers=layers)

    # loss
    loss_function = torch.nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 开始训练
    model.train()
    for i in range(epochs):
        training_loss = 0
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq)  # 压缩维度：得到输出，并将维度为1的去除
            single_loss = loss_function(y_pred, labels)
            training_loss += single_loss.item() * len(seq)
            single_loss.backward()
            optimizer.step()
        print("epoch {} Training loss: {}".format(i, training_loss / len(train_loader.dataset)))

    # 开始验证
    valid_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x_valid, y_valid),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=64,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多进程（multiprocess）来读数据
    )

    # 开始验证
    model.eval()
    correct = 0.
    loss = 0.
    valid_preds = []
    valid_labels = []
    for seq, labels in valid_loader:  # 这里偷个懒，就用训练数据验证哈！
        y_pred = model(seq)  # 压缩维度：得到输出，并将维度为1的去除
        single_loss = loss_function(y_pred, labels)

        pred = y_pred.argmax(dim=1)
        label = labels.argmax(dim=1)
        valid_preds.append(pred)
        valid_labels.append(label)

        correct += int((pred == label).sum())
        loss += single_loss * len(seq)

    val_acc = correct / len(valid_loader.dataset)
    val_loss = loss / len(valid_loader.dataset)
    print("\nnormal Validation loss: {}\taccuracy:{}".format(val_loss, val_acc))

    acc, recall, precision, f1, total_num = calculate_metrics(valid_preds, valid_labels)
    print("total:{} \n结果指标\tacc:{} recall:{} precision:{} f1:{}".format(total_num, acc, recall, precision, f1))

    # 得到数据
    x, y = get_etherscan_data()
    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x, y),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=64,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多进程（multiprocess）来读数据
    )

    # 开始测试
    model.eval()
    loss = 0.
    correct = 0.
    test_preds = []
    test_labels = []
    for seq, labels in test_loader:  # 这里偷个懒，就用训练数据验证哈！
        y_pred = model(seq)  # 压缩维度：得到输出，并将维度为1的去除
        single_loss = loss_function(y_pred, labels)

        pred = y_pred.argmax(dim=1)
        label = labels.argmax(dim=1)
        test_preds.append(pred)
        test_labels.append(label)

        correct += int((pred == label).sum())
        loss += single_loss * len(seq)

    val_acc = correct / len(test_loader.dataset)
    val_loss = loss / len(test_loader.dataset)
    print("\nnormal Validation loss: {}\taccuracy:{}".format(val_loss, val_acc))

    acc, recall, precision, f1, total_num = calculate_metrics(test_preds, test_labels)
    print("total:{} \n结果指标\tacc:{} recall:{} precision:{} f1:{}".format(total_num, acc, recall, precision, f1))
