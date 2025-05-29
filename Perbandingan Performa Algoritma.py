import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze()

# Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)
acc_dt = accuracy_score(y_test, dt_model.predict(X_test))

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
acc_rf = accuracy_score(y_test, rf_model.predict(X_test))

# Pie Chart Akurasi
plt.figure(figsize=(6, 6))
plt.pie([acc_dt, acc_rf],
        labels=['Decision Tree', 'Random Forest'],
        autopct='%1.1f%%',
        colors=['#66c2a5', '#fc8d62'],
        startangle=140)
plt.title('Perbandingan Akurasi Model')
plt.tight_layout()
plt.show()
