import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("heart.csv")

print("Dataset loaded successfully")
print("Dataset shape:", data.shape)


# ==========================================
# 2. REMOVE DUPLICATES
# ==========================================

data = data.drop_duplicates()

print("After removing duplicates:", data.shape)


# ==========================================
# 3. SEPARATE INPUT AND TARGET
# ==========================================

X = data.drop("target", axis=1)
target_labels = set(data["target"].unique())
if target_labels != {0, 1}:
    raise ValueError(
        f"Expected binary target labels 0 and 1, got {sorted(target_labels)}"
    )

# In this dataset, target 0 has the stronger disease-associated feature
# patterns, so map it to the app's positive (disease) class.
y = data["target"].map({0: 1, 1: 0})


# ==========================================
# 4. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 5. RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    random_state=42
)


# ==========================================
# 6. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

print("Random Forest model trained successfully")


# ==========================================
# 7. CHECK ACCURACY
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


# ==========================================
# 8. TRAIN FINAL MODEL ON COMPLETE DATA
# ==========================================

model.fit(X, y)


# ==========================================
# 9. SAVE MODEL
# ==========================================

joblib.dump(model, "model_joblib_heart.pkl")

print("Model saved successfully!")
print("File: model_joblib_heart.pkl")