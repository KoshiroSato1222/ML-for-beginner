import pandas as pd
# 機械学習のパッケージをインポート
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = pd.read_csv("iris.data", names=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"])

# 入力データと正解データに分類する（x, y)
y = iris.loc[:, "Class"]
x = iris.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

# 学習用データとテストデータに分離する
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)

# 学習する
clf = SVC()
clf.fit(x_train, y_train)

# テストを評価する、機械学習した時の予測結果
y_pred = clf.predict(x_test)

# 答え合わせして正解率を表示する
print("正解率は？？", accuracy_score(y_test, y_pred))
