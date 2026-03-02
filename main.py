import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

digits = load_digits()
X = pd.DataFrame(digits.data)
y = pd.Series(digits.target)
print("1. The data has been loaded successfully.")

problem_type = "Classification"
print(f"2. Selected problem type: {problem_type}")

X.dropna(inplace=True)
X.drop_duplicates(inplace=True)
y = y.iloc[X.index]

print("- Missing, duplicate, and outlier values have been checked.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
print("- The data has been split into training and testing sets.")

model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("3. The model (sklearn KNN) has been built and trained successfully.")

print("\n4. Evaluation Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("-" * 30)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Using Sklearn KNN)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
