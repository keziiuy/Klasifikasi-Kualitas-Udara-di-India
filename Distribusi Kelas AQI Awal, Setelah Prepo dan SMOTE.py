import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_awal = 'city_day_clean.csv'
file_smote = 'city_day_balanced_SMOTE.csv'

target = 'AQI_Bucket'

df_awal = pd.read_csv(file_awal)
df_smote = pd.read_csv(file_smote)

# Hitung distribusi
dist_awal = df_awal[target].value_counts().sort_index()
dist_smote = df_smote[target].value_counts().sort_index()

df_compare = pd.DataFrame({
    'Dataset Awal': dist_awal,
    'Setelah Prepo & SMOTE': dist_smote
}).reset_index().rename(columns={'index': 'AQI_Bucket'})

# Plot perbandingan
plt.figure(figsize=(10, 6))
bar_width = 0.4
x = range(len(df_compare))

plt.bar([i - bar_width/2 for i in x], df_compare['Dataset Awal'], width=bar_width, label='Dataset Awal', color='skyblue')
plt.bar([i + bar_width/2 for i in x], df_compare['Setelah Prepo & SMOTE'], width=bar_width, label='Setelah Prepo & SMOTE', color='salmon')

plt.xticks(ticks=x, labels=df_compare['AQI_Bucket'])
plt.xlabel('AQI_Bucket')
plt.ylabel('Jumlah')
plt.title('Distribusi Kelas AQI: Awal vs Setelah Preprocessing & SMOTE')
plt.legend()
plt.tight_layout()
plt.show()
