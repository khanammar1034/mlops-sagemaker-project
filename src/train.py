from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading dataset...")

iris = load_iris()

X = iris.data
y = iris.target

print("Training model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

print("Saving model...")

joblib.dump(model, "model.joblib")

print("Training complete!")