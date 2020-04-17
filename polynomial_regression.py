# Import libraries.
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Defining input and output datapoints.
x = np.array([3, 6, 12, 24, 48, 96]).reshape((-1, 1))
y = np.array([14, 10, 5, 9, 35, 42])

# Transforming input data points.
transformer = PolynomialFeatures(degree=2, include_bias=False)
transformer.fit(x)
x_ = transformer.transform(x)
x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
print(x_)


# Creating a multi-linear regression model.
model = LinearRegression().fit(x_, y)

# Computing result. 
r_sq = model.score(x_, y)
print('coefficient of determination:', r_sq)
print("\n")
print('intercept:', model.intercept_)
print("\n")
print('coefficients:', model.coef_)

# Making prediction.
x_ = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)
print(x_)

