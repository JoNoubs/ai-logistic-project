import os
from dotenv import load_dotenv
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load environment variables
load_dotenv()

# Use environment variables
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.01'))
EPOCHS = int(os.getenv('EPOCHS', '100'))

print(f"Using Learning Rate: {LEARNING_RATE}")
print(f"Number of Epochs: {EPOCHS}")

# Load and prepare data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Implement logistic regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(X, y, learning_rate, epochs):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    
    for _ in range(epochs):
        z = np.dot(X, weights) + bias
        predictions = sigmoid(z)
        
        dw = (1/m) * np.dot(X.T, (predictions - y))
        db = (1/m) * np.sum(predictions - y)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
    
    return weights, bias

# Train the model
weights, bias = train(X_train, y_train, LEARNING_RATE, EPOCHS)

# Evaluate the model
def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)

y_pred = predict(X_test, weights, bias)
accuracy = np.mean((y_pred >= 0.5) == y_test)

print(f"Model accuracy: {accuracy}")
