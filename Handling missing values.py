import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

df = pd.read_csv('city_day.csv')

# Drop kolom waktu karena non-time series
df.drop(columns=['Date'], inplace=True)

# Identifikasi kolom numerik dan kategorik
numerik = df.select_dtypes(include=[np.number]).columns.tolist()
kategorikal = df.select_dtypes(include=['object']).columns.tolist()

# Imputasi missing values numerik dengan median
imputer_num = SimpleImputer(strategy='median')
df[numerik] = imputer_num.fit_transform(df[numerik])

# Imputasi missing values kategorik dengan mode
imputer_cat = SimpleImputer(strategy='most_frequent')
df[kategorikal] = imputer_cat.fit_transform(df[kategorikal])

df.to_csv('city_day_clean.csv', index=False)
print("File untuk eksplorasi (EDA) disimpan sebagai 'city_day_clean.csv'")

print("Missing values setelah imputasi:")
print(df.isnull().sum())
