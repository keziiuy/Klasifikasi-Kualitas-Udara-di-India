import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datasets = {
    'city_day_clean.csv': 'Dataset Awal (Sebelum Seleksi Fitur)',
    'city_day_selected_features.csv': 'Setelah Seleksi Fitur, Sebelum Outlier',
    'city_day_no_outliers.csv': 'Setelah Outlier Dihapus, Sebelum SMOTE',
    'city_day_balanced_SMOTE.csv': 'Setelah SMOTE (Data Seimbang)'
}

# Target kolom tetap
target = 'AQI_Bucket'

# Fitur numerik pendek untuk 3 file terakhir
fitur_numerik_default = ['PM2.5', 'PM10', 'CO', 'O3', 'NO', 'NO2', 'SO2', 'NOx']

for file, judul_kondisi in datasets.items():
    print(f"\n====== {judul_kondisi} ======\n")
    df = pd.read_csv(file)

    if file == 'city_day_clean.csv':
        numerik_fitur = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                         'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
    else:
        numerik_fitur = fitur_numerik_default

    print("Descriptive Statistics:")
    print(df[numerik_fitur].describe())

    # Histogram fitur numerik
    plt.figure(figsize=(14, 10))
    df[numerik_fitur].hist(bins=20, figsize=(14, 10), color='skyblue', edgecolor='black')
    plt.suptitle(f'Distribusi Fitur Numerik ({judul_kondisi})', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Barplot distribusi kelas AQI
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target, data=df, palette='Set2')
    plt.title(f'Distribusi Kelas AQI ({judul_kondisi})')
    plt.xlabel('Kategori AQI')
    plt.ylabel('Jumlah')
    plt.tight_layout()
    plt.show()

    # Countplot kota per kategori AQI
    plt.figure(figsize=(14, 6))
    if 'City' in df.columns:
        sns.countplot(x='City', hue=target, data=df, palette='Set3')
        plt.title(f'Distribusi AQI_Bucket Berdasarkan Kota ({judul_kondisi})')
        plt.xlabel('Kota')
        plt.ylabel('Jumlah')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        print("Kolom 'City' tidak ditemukan, skip visualisasi berdasarkan kota.")

    # Boxplot setiap fitur terhadap kategori AQI
    plt.figure(figsize=(14, 10))
    for i, feature in enumerate(numerik_fitur, 1):
        plt.subplot((len(numerik_fitur) - 1) // 3 + 1, 3, i)
        sns.boxplot(x=target, y=feature, data=df, palette='Pastel1')
        plt.title(f'{feature} vs {target}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
