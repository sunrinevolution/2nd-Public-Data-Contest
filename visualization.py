import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

foreign_student = pd.read_csv('./data/foreign_student.csv')

foreign_student = foreign_student.reindex(sorted(foreign_student.columns), axis=1)

print(foreign_student.info())

h1n1 = pd.read_csv('./data/H1N1_preprocess.csv')
mers = pd.read_csv('./data/mers_preprocess.csv')

h1n1 = h1n1.drop(columns=[h1n1.columns[0]])
mers = mers.drop(columns=[mers.columns[0]])

print(h1n1)

print(mers)