import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('city_day_clean.csv')  

le_city = LabelEncoder()
df['City_Encoded'] = le_city.fit_transform(df['City'])

fitur = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO', 'SO2',
         'O3', 'Benzene', 'Toluene', 'Xylene', 'NH3', 'City_Encoded']
X = df[fitur]
y = df['AQI_Bucket']  

# Latih model Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Hitung dan urutkan feature importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Fitur': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Urutan pentingnya fitur:")
print(feature_importance_df)

# Pilih fitur dengan importance > 0.05
selected_features = feature_importance_df[feature_importance_df['Importance'] > 0.05]['Fitur'].tolist()
print(f"Fitur yang dipilih untuk modeling: {selected_features}")

selected_df = df[selected_features + ['AQI_Bucket', 'City']]
selected_df.to_csv('city_day_selected_features.csv', index=False)
print("File dengan fitur terpilih berhasil disimpan sebagai 'city_day_selected_features.csv'")
