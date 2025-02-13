import os
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

load_dotenv()

# Use environment variables
test_size = float(os.getenv('TEST_SIZE', '0.2'))

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
print(f"Model accuracy: {model.score(X_test, y_test)}")