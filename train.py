from matplotlib.pyplot import axes
import pandas as pd
from fbprophet import Prophet
from pandas.core.frame import DataFrame

raw_data = pd.read_csv('./rawdata/student.csv', encoding='CP949')

raw_data = raw_data.fillna(0)

data = pd.DataFrame(raw_data.sum()) # 유학생 데이터

print(data.head(10))

all_data = pd.read_csv('./data/all_preprocess.csv')
all_data = all_data.drop(columns=list(all_data.columns)[0])

print(all_data)