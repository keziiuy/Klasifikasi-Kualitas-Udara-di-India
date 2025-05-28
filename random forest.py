import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze()

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred = rf_model.predict(X_test)

# Evaluation
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Akurasi: {accuracy_score(y_test, y_pred):.4f}")

# Visualisasi Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.tight_layout()
plt.show()

# Visualisasi salah satu pohon representatif dari Random Forest
# Ambil 1 estimator 
chosen_tree = rf_model.estimators_[0]

# Visualisasi pohon dengan pembatasan kedalaman
plt.figure(figsize=(24, 12))
plot_tree(chosen_tree,
          feature_names=X_train.columns,
          class_names=[str(cls) for cls in rf_model.classes_],
          filled=True, rounded=True, max_depth=3, fontsize=10)
plt.title('Visualisasi Salah Satu Pohon (Depth ≤ 3) dari Random Forest')
plt.tight_layout()
plt.show()
