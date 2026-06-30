import os
import joblib

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

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

# -----------------------------
# Determine where to save model
# -----------------------------

model_dir = os.environ.get("SM_MODEL_DIR", ".")

os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.joblib")

print(f"Saving model to: {model_path}")

joblib.dump(model, model_path)

print("Training complete!")