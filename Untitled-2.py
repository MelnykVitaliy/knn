import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# 1. Генеруємо випадкові дані
np.random.seed(42)
X = np.sort(np.random.rand(1000, 1) * 10, axis=0)  # X: від 0 до 10
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])  # y: синус + шум

# 2. Нормалізація
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Розділення на навчальну і тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 4. Навчання KNN-регресора з різними K
k_range = range(1, 21)
mse_scores = []

for k in k_range:
    knn_reg = KNeighborsRegressor(n_neighbors=k)
    knn_reg.fit(X_train, y_train)
    y_pred = knn_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

# 5. Вибір найкращого K
best_k = k_range[np.argmin(mse_scores)]
best_mse = min(mse_scores)
print(f"Найкраще K = {best_k}, MSE = {best_mse:.4f}")

# 6. Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(k_range, mse_scores, marker='o')
plt.xlabel('Значення K')
plt.ylabel('Середньоквадратична помилка (MSE)')
plt.title('KNN-регресор: вибір найкращого K')
plt.grid(True)
plt.show()

# 7. Побудова лінії передбачення для найкращого K
knn_best = KNeighborsRegressor(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

# Візуалізація передбачення
X_all_sorted = np.sort(X_scaled, axis=0)
y_pred_all = knn_best.predict(X_all_sorted)

plt.figure(figsize=(10, 6))
plt.scatter(X_scaled, y, c='lightgray', label='Вихідні дані')
plt.plot(X_all_sorted, y_pred_all, color='blue', linewidth=2, label=f'KNN передбачення (k={best_k})')
plt.title('KNN-регресор: передбачення')
plt.xlabel('X (нормалізований)')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
