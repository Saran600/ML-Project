from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("saved_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
