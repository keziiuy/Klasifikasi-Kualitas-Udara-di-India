import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('city_day_selected_features.csv')

numerik_fitur = ['PM2.5', 'PM10', 'CO', 'O3', 'NO', 'NO2', 'SO2', 'NOx']

# Fungsi untuk mendeteksi dan menghapus outliers berdasarkan IQR
def remove_outliers(df, features):
    df_out = df.copy()
    for feature in features:
        Q1 = df_out[feature].quantile(0.25)
        Q3 = df_out[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_count = df_out[(df_out[feature] < lower_bound) | (df_out[feature] > upper_bound)].shape[0]
        print(f"Fitur {feature}: menghapus {outlier_count} outlier")

        df_out = df_out[(df_out[feature] >= lower_bound) & (df_out[feature] <= upper_bound)]
    return df_out

# Menangani outliers
df_cleaned = remove_outliers(df, numerik_fitur)

# Menampilkan jumlah data sebelum dan sesudah menghapus outliers
print(f"\nJumlah data sebelum menghapus outliers: {len(df)}")
print(f"Jumlah data setelah menghapus outliers: {len(df_cleaned)}")

# Tampilkan statistik deskriptif setelah menghapus outliers
print("\n===== Statistik Deskriptif Setelah Menghapus Outliers =====")
print(df_cleaned[numerik_fitur].describe())

# Visualisasi distribusi fitur numerik setelah menghapus outliers
plt.figure(figsize=(14, 10))
df_cleaned[numerik_fitur].hist(bins=20, figsize=(14, 10), color='skyblue', edgecolor='black')
plt.suptitle('Distribusi Fitur Numerik setelah Menghapus Outliers', fontsize=16)
plt.tight_layout()
plt.show()

df_cleaned.to_csv('city_day_no_outliers.csv', index=False)
