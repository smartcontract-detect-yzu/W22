from numpy import loadtxt
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# dataset_path = "../dataset/dataset_001.csv"
dataset_path = "xgboost_dataset_all.csv"
dataset = loadtxt(dataset_path, delimiter=",")
X = dataset[:, 1:]
Y = dataset[:, 0]

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# 不可视化数据集loss
# model = XGBClassifier()
# model.fit(X_train, y_train)

##可视化测试集的loss
model = XGBClassifier()
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=False)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]



accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
##Accuracy: 77.56%


valid_dataset_path = "xgboost_dataset_etherscan.csv"
valid_dataset = loadtxt(valid_dataset_path, delimiter=",")
X_valid = valid_dataset[:, 1:]
Y_valid = valid_dataset[:, 0]

y_valid_pred = model.predict(X_valid)
predictions = [round(value) for value in y_valid_pred]

accuracy = accuracy_score(Y_valid, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

from matplotlib import pyplot
plot_importance(model)
pyplot.show()