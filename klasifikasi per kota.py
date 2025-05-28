import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('city_day_balanced_SMOTE.csv')

features = ['PM2.5', 'PM10', 'CO', 'O3', 'NO', 'NO2', 'SO2', 'NOx']
target = 'AQI_Bucket'

X = df[features]
y = df[target]

# Buat dan latih model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Prediksi label semua data
df['Predicted_Label'] = rf_model.predict(X)

# Ambil label dominan per kota (mode)
city_label = df.groupby('City')['Predicted_Label'].agg(lambda x: x.mode()[0]).reset_index()

city_label.to_csv('city_classification_result.csv', index=False)
print("Hasil klasifikasi per kota berhasil disimpan di 'city_classification_result.csv'")
