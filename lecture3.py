import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost.sklearn import XGBClassifier

data = pd.read_csv("diabetes.csv")

# x, yに分割する
x = data.drop(["Outcome"], axis=1)
y = data["Outcome"]

# 学習用データとテストデータに分離する
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)


# # 学習する
# xgb = XGBClassifier()
# xgb.fit(x_train, y_train)
#
# # テストを評価する、機械学習した時の予測結果
# y_pred = xgb.predict(x_test)
#
# # 答え合わせして正解率を表示する
# print("正解率は？？", accuracy_score(y_test, y_pred))

# パラメータの検索（パラメータチューニング）
# パラメータの選択肢を範囲で指定
params = {"eta": [0.1, 0.3, 0.9], 'max_depth': [2, 4, 4, 8]}
xgb_grid = GridSearchCV(
    estimator=XGBClassifier(),
    param_grid=params
)

xgb_grid.fit(x_train, y_train)

for key, value in xgb_grid.best_params_.items():
    print(key, value)

y_pred = xgb_grid.predict(x_test)
print("正解率は？？", accuracy_score(y_test, y_pred))