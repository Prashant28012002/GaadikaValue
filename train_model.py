import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

BASE_DIR = r"D:\Vehicle_project"
DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")

data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip().str.lower()

# Clean price
data["price"] = (
    data["price"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("$", "", regex=False)
)
data["price"] = pd.to_numeric(data["price"], errors="coerce")
data.dropna(subset=["price"], inplace=True)

# Targets
y_price = data["price"]
y_make = data["make"]

# Features
X = data.drop("price", axis=1)

num_features = ["year", "mileage", "cylinders", "doors"]
cat_features = [
    "make", "model", "fuel", "transmission",
    "body", "drivetrain", "exterior_color",
    "interior_color", "trim"
]

num_features = [c for c in num_features if c in X.columns]
cat_features = [c for c in cat_features if c in X.columns]

# Pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_features),
    ("cat", categorical_pipeline, cat_features)
])

# PRICE MODEL
price_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X, y_price, test_size=0.2, random_state=42
)

price_model.fit(X_train_p, y_train_p)

print("Price MAE:", mean_absolute_error(y_test_p, price_model.predict(X_test_p)))
print("Price R2:", r2_score(y_test_p, price_model.predict(X_test_p)))

# BRAND MODEL
X_make = X.drop("make", axis=1)
cat_features_m = [c for c in cat_features if c != "make"]

preprocessor_make = ColumnTransformer([
    ("num", numeric_pipeline, num_features),
    ("cat", categorical_pipeline, cat_features_m)
])

brand_model = Pipeline([
    ("preprocessor", preprocessor_make),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_make, y_make, test_size=0.2, random_state=42
)

brand_model.fit(X_train_m, y_train_m)

print("Brand Accuracy:", accuracy_score(y_test_m, brand_model.predict(X_test_m)))

# Save models
joblib.dump(price_model, os.path.join(BASE_DIR, "price_model.joblib"))
joblib.dump(brand_model, os.path.join(BASE_DIR, "brand_model.joblib"))

print("Models saved successfully ✅")
