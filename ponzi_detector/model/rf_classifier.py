import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

# seed = 2
# test_size = 0.35


seed = 2
test_size = 0.35

# dataset_path = "../dataset/dataset_001.csv"
ponzi_dataset_path = "xgboost_dataset_all_ponzi.csv"
ponzi_dataset = loadtxt(ponzi_dataset_path, delimiter=",")
X_p = ponzi_dataset[:, 1:]
Y_p = ponzi_dataset[:, 0]
X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(X_p, Y_p, test_size=test_size, random_state=seed)

no_ponzi_dataset_path = "xgboost_dataset_all_no_ponzi.csv"
no_ponzi_dataset = loadtxt(no_ponzi_dataset_path, delimiter=",")
X_np = no_ponzi_dataset[:, 1:]
Y_np = no_ponzi_dataset[:, 0]
X_np_train, X_np_test, y_np_train, y_np_test = train_test_split(X_np, Y_np, test_size=test_size, random_state=seed)

X_train = np.concatenate((X_p_train, X_np_train))
y_train = np.concatenate((y_p_train, y_np_train))

X_test = np.concatenate((X_p_test, X_np_test))
y_test = np.concatenate((y_p_test, y_np_test))

# 训练集
clf = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0)
clf = clf.fit(X_train, y_train)
rfc = rfc.fit(X_train, y_train)

# 验证集结果
score_c = clf.score(X_test, y_test)
score_r = rfc.score(X_test, y_test)

y_pred = rfc.predict(X_test)
print("\n大数据集结果：acc:{} recall:{} pre:{} f1:{}".format(
    score_r,
    recall_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    f1_score(y_test, y_pred)))

valid_dataset_path = "xgboost_dataset_etherscan.csv"
valid_dataset = loadtxt(valid_dataset_path, delimiter=",")
X_valid = valid_dataset[:, 1:]
Y_valid = valid_dataset[:, 0]

score_c = clf.score(X_valid, Y_valid)
score_r = rfc.score(X_valid, Y_valid)

# 测试集结果
y_pred = rfc.predict(X_valid)
print("\n小数据集结果：acc:{} recall:{} pre:{} f1:{}".format(
    score_r,
    recall_score(Y_valid, y_pred),
    precision_score(Y_valid, y_pred),
    f1_score(Y_valid, y_pred)))
