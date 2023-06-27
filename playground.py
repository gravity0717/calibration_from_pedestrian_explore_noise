import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# 입력 변수 생성
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

# 출력 변수 생성
y = np.array([2, 4, 6, 8, 10])

# 다항식 변환
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

# 다항 회귀 모델 학습
model = LinearRegression()
model.fit(X_poly, y)

# 예측을 위한 새로운 데이터 생성
X_new = np.array([6]).reshape(-1, 1)
X_new_poly = poly_features.transform(X_new)

# 예측
y_new = model.predict(X_new_poly)

# 결과 출력
print("예측값:", y_new)

# 다항 회귀 모델 시각화
X_plot = np.linspace(1, 6, 100).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)
y_plot = model.predict(X_plot_poly)

plt.scatter(X, y, label='Actual')
plt.plot(X_plot, y_plot, color='red', label='Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
