import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

df = pd.read_csv('city_day_no_outliers.csv')

features = ['PM2.5', 'PM10', 'CO', 'O3', 'NO', 'NO2', 'SO2', 'NOx']
target = 'AQI_Bucket'

X = df[features]
y = df[target]

city_series = df['City']

print("Distribusi sebelum SMOTE:", Counter(y))

# Visualisasi sebelum SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette='Set2')
plt.title('Distribusi Kelas AQI Sebelum SMOTE')
plt.xlabel('AQI Bucket')
plt.ylabel('Jumlah')
plt.tight_layout()
plt.show()

# Split data untuk SMOTE
X_train, X_test, y_train, y_test, city_train, _ = train_test_split(
    X, y, city_series, test_size=0.2, random_state=42, stratify=y
)

city_train = city_train.reset_index(drop=True)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Visualisasi setelah SMOTE
print("Distribusi setelah SMOTE:", Counter(y_train_resampled))

plt.figure(figsize=(6, 4))
sns.countplot(x=y_train_resampled, palette='Set2')
plt.title('Distribusi Kelas AQI Setelah SMOTE')
plt.xlabel('AQI Bucket')
plt.ylabel('Jumlah')
plt.tight_layout()
plt.show()

resampled_df = pd.DataFrame(X_train_resampled, columns=features)
resampled_df[target] = y_train_resampled

repeated_city = city_train.sample(n=len(resampled_df), replace=True, random_state=42).reset_index(drop=True)
resampled_df['City'] = repeated_city

resampled_df.to_csv('city_day_balanced_SMOTE.csv', index=False)
print("Data hasil SMOTE dengan kolom City disimpan ke city_day_balanced_SMOTE.csv")
