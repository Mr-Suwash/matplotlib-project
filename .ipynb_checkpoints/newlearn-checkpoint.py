from sklearn.datasets import load_boston
import pandas as pd

load_boston()
df=pd.read_csv('C:/Users/DELL/Desktop/layoffs.csv')
print(df.head())