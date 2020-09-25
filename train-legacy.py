import numpy as np # 행렬 연산 / 데이터 핸들링
import pandas as pd # 데이터 분석
import matplotlib.pyplot as plt # 그래프 시각화
from xgboost import XGBRegressor # XGBoost Regressor 모델
from sklearn.model_selection import KFold # K-validation
from sklearn.metrics import accuracy_score # 정확도 측정 함수
from sklearn.preprocessing import LabelEncoder # 라벨 인코더

X_dataframe = pd.read_csv('./data/H1N1_preprocess.csv')
y_dataframe = pd.read_csv('./data/foreign_student.csv')

print(y_dataframe)

X_train = np.array(X_dataframe)
y_train = np.array(y_dataframe)

# X_train = pd.
# # y_train = pd.read_csv('./data/foreign_student.csv')

# rf = RandomForestRegressor(random_state=1217)  # RandomForestRegressor 생성

# rf.fit(X_train, y_train) #생성된 트리모델을 학습시키는 과정

# n_feature = X_train.shape[1] #주어진 변수들의 갯수를 구함
# index = np.arange(n_feature)

# a = rf.predict(x_val)

model = XGBRegressor(random_state=110, verbosity=0, nthread=23, n_estimators=980, max_depth=4)
kfold = KFold(n_splits=8, shuffle=True, random_state=777)
n_iter = 0
cv_score = []

def rmse(target, pred):
  return np.sqrt(np.sum(np.power(target - pred, 2)) / np.size(pred))

for train_index, test_index in kfold.split(X_train, y_train):
  # K Fold가 적용된 train, test 데이터를 불러온다
  # X_train, X_test = X_train.iloc[train_index,:], X_train.iloc[test_index, :]
  # Y_train, Y_test = y_train.iloc[train_index], y_train.iloc[test_index]
  
  # 모델 학습과 예측 수행
  model.fit(X_train, y_train)
  # pred = model.predict(X_test)
  # print(pred)
  
  # 정확도 RMSE 계산
  n_iter += 1
  # score = rmse(Y_test, pred)
  print(score)
  cv_score.append(score)

print('\n교차 검증별 RMSE :', np.round(cv_score, 4))
print('평균 검증 RMSE :', np.mean(cv_score))