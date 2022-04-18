from numpy import loadtxt
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np

seed = 4
test_size = 0.3

print("{} {}".format(seed, test_size))

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
