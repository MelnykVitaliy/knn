import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

df = pd.concat([X, y.rename("target")], axis=1).sample(frac=1, random_state=42).reset_index(drop=True)
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

scores = []
k_range = range(1, 21)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

plt.plot(k_range, scores, marker='o')
plt.xlabel('Значення K')
plt.ylabel('Точність (Accuracy)')
plt.title('KNN: вибір найкращого K для Iris dataset')
plt.grid(True)
plt.show()

best_k = k_range[np.argmax(scores)]
best_score = max(scores)
print(f"Найкраще K = {best_k}, точність = {best_score:.2f}")
