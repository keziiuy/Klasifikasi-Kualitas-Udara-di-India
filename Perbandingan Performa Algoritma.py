import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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

print(f"Akurasi Decision Tree: {acc_dt:.4f}")
print(f"Akurasi Random Forest: {acc_rf:.4f}")

# Visualisasi barplot akurasi
plt.figure(figsize=(6, 4))
sns.barplot(x=['Decision Tree', 'Random Forest'], y=[acc_dt, acc_rf], palette='Set2')
plt.title('Perbandingan Akurasi Model')
plt.ylabel('Akurasi')
plt.ylim(0, 1)
for i, acc in enumerate([acc_dt, acc_rf]):
    plt.text(i, acc + 0.02, f"{acc:.3f}", ha='center', fontsize=10)
plt.tight_layout()
plt.show()
