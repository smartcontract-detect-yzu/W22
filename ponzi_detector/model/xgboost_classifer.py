from matplotlib import pyplot as plt
from numpy import loadtxt
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np

# seed = 4
# test_size = 0.3

OPCODE_MAP = {
    "STOP": 1,
    "ADD": 2,
    "SUB": 3,
    "MUL": 4,
    "DIV": 5,
    "SDIV": 6,
    "MOD": 7,
    "SMOD": 8,
    "EXP": 9,
    "NOT": 10,
    "LT": 11,
    "GT": 12,
    "SLT": 13,
    "SGT": 14,
    "EQ": 15,
    "ISZERO": 16,
    "AND": 17,
    "OR": 18,
    "XOR": 19,
    "BYTE": 20,
    "SHL": 21,
    "SHR": 22,
    "SAR": 23,
    "ADDMOD": 24,
    "MULMOD": 25,
    "SIGNEXTEND": 26,
    "KECCAK256": 27,
    "ADDRESS": 28,
    "BALANCE": 29,
    "ORIGIN": 30,
    "CALLER": 31,
    "CALLVALUE": 32,
    "CALLDATALOAD": 33,
    "CALLDATASIZE": 34,
    "CALLDATACOPY": 35,
    "CODESIZE": 36,
    "CODECOPY": 37,
    "GASPRICE": 38,
    "EXTCODESIZE": 39,
    "EXTCODECOPY": 40,
    "RETURNDATASIZE": 41,
    "RETURNDATACOPY": 42,
    "EXTCODEHASH": 43,
    "BLOCKHASH": 44,
    "COINBASE": 45,
    "TIMESTAMP": 46,
    "NUMBER": 47,
    "DIFFICULTY": 48,
    "GASLIMIT": 49,
    "CHAINID": 50,
    "SELFBALANCE": 51,
    "POP": 52,
    "MLOAD": 53,
    "MSTORE": 54,
    "MSTORE8": 55,
    "SLOAD": 56,
    "SSTORE": 57,
    "JUMP": 58,
    "JUMPI": 59,
    "PC": 60,
    "MSIZE": 61,
    "GAS": 62,
    "JUMPDEST": 63,
    "LOG0": 64,
    "LOG1": 65,
    "LOG2": 66,
    "LOG3": 67,
    "LOG4": 68,
    "CREATE": 69,
    "CALL": 70,
    "CALLCODE": 71,
    "STATICCALL": 72,
    "RETURN": 73,
    "DELEGATECALL": 74,
    "CREATE2": 75,
    "REVERT": 76,
    "INVALID": 77,
    "SELFDESTRUCT": 78
}

seed = 4
test_size = 0.33

print("{} {}".format(seed, test_size))

# dataset_path = "../dataset/dataset_001.csv"
ponzi_dataset_path = "xgboost_dataset_all_ponzi.csv"
ponzi_dataset = loadtxt(ponzi_dataset_path, delimiter=",")
X_p = ponzi_dataset[:, 1:]
Y_p = ponzi_dataset[:, 0]
X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(X_p, Y_p, test_size=test_size, random_state=seed,
                                                            shuffle=True)

no_ponzi_dataset_path = "xgboost_dataset_all_no_ponzi.csv"
no_ponzi_dataset = loadtxt(no_ponzi_dataset_path, delimiter=",")
X_np = no_ponzi_dataset[:, 1:]
Y_np = no_ponzi_dataset[:, 0]
X_np_train, X_np_test, y_np_train, y_np_test = train_test_split(X_np, Y_np, test_size=test_size, random_state=seed,
                                                                shuffle=True)

X_train = np.concatenate((X_p_train, X_np_train))
y_train = np.concatenate((y_p_train, y_np_train))
# X_train = X_p_train + X_np_train
# y_train = y_p_train + y_np_train


X_test = np.concatenate((X_p_test, X_np_test))
y_test = np.concatenate((y_p_test, y_np_test))
# X_test = X_p_test + X_np_test
# y_test = y_p_test + y_np_test

# 不可视化数据集loss
# model = XGBClassifier()
# model.fit(X_train, y_train)

##可视化测试集的loss
model = XGBClassifier(learning_rate=0.01, objective='binary:logistic', seed=27)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=False)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
print("\n大数据集结果：acc:{} recall:{} pre:{} f1:{}".format(accuracy, recall, precision, f1))
print("Accuracy: %.2f%%" % (accuracy * 100.0))


smaple_path = "xgboost_feature.txt"
sample = loadtxt(smaple_path, delimiter=",")
X_s = sample[:, 1:]
Y_s = sample[:, 0]
y_pred = model.predict(X_s)
score_r = model.score(X_s, Y_s)
print("11111 :{} {}".format(y_pred, score_r))


valid_dataset_path = "xgboost_dataset_etherscan.csv"
valid_dataset = loadtxt(valid_dataset_path, delimiter=",")
X_valid = valid_dataset[:, 1:]
Y_valid = valid_dataset[:, 0]

y_valid_pred = model.predict(X_valid)
predictions = [round(value) for value in y_valid_pred]

accuracy = accuracy_score(Y_valid, predictions)
recall = recall_score(Y_valid, predictions)
precision = precision_score(Y_valid, predictions)
f1 = f1_score(Y_valid, predictions)
print("\n小数据集结果：acc:{} recall:{} pre:{} f1:{}".format(accuracy, recall, precision, f1))
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# from matplotlib import pyplot
# plot_importance(model)
# pyplot.show()

importances = model.feature_importances_
# print("model.feature_importances_: {}".format(importances))

indices = np.argsort(importances)[::-1]
top_k = indices[0:5]
# print(indices[0:10])

feature_names = {}
for name in OPCODE_MAP:
    id = OPCODE_MAP[name]
    feature_names[id] = name

# 获取特征名字
names = [feature_names[i + 1] for i in top_k]

# # 创建图
plt.figure()
plt.title("Feature Importance Of RF")
# features.shape[1]  数组的长度
# plt.bar(range(10), importances[top_10])
# plt.xticks(range(10), names, rotation=90)

plt.bar(range(len(top_k)), importances[top_k])
plt.xticks(range(len(top_k)), names)
plt.show()
