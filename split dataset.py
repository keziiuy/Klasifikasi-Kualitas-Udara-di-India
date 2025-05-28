import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('city_day_balanced_SMOTE.csv')

features = ['PM2.5', 'PM10', 'CO', 'O3', 'NO', 'NO2', 'SO2', 'NOx']
target = 'AQI_Bucket'

X = df[features]
y = df[target]

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Jumlah data total: {len(df)}")
print(f"Train set: {len(X_train)} sampel")
print(f"Test set: {len(X_test)} sampel")

X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Data train dan test telah disimpan sebagai X_train.csv, X_test.csv, y_train.csv, y_test.csv")
