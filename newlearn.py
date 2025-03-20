from sklearn.datasets import fetch_california_housing  # No parentheses while importing

# Load dataset
housing = fetch_california_housing()

# Print dataset description
print(housing)
# Separate the features (X) and target (y)
X = housing.data
y = housing.target

# Print the shapes of X and y
print(X.shape)
print(y.shape)

from sklearn.neighbors import KNeighborsRegressor
mod = KNeighborsRegressor()
mod.fit(X,y)
mod.predict(X)