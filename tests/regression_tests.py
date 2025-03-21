import numpy as np
import unittest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Generate synthetic dataset
np.random.seed(42)
X = np.random.rand(100, 1) * 10  
y = 3 * X.squeeze() + 7 + np.random.randn(100) * 2 

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Test with a known past value
class TestRegression(unittest.TestCase):
    def test_regression_prediction(self):
        test_input = np.array([[5.0]])  
        expected_output = 3 * 5.0 + 7  
        predicted_output = model.predict(test_input)[0]
        self.assertAlmostEqual(predicted_output, expected_output, delta=2.0)  # Allow noise variance
