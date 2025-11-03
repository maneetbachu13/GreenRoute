import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ===============================
# 🌿 1. Load Training Data
# ===============================

if not os.path.exists("training_data.csv"):
    print("❌ training_data.csv not found! Please generate some route data first.")
    exit()

# Load CSV (no header in our case)
df = pd.read_csv("training_data.csv", header=None)
df.columns = [
    "timestamp",
    "distance_km",
    "elevation_gain_m",
    "congestion_index",
    "fuel_type",
    "base_emission_per_km",
    "emission",
    "aqi"
]

# Drop missing or invalid data
df = df.dropna(subset=["distance_km", "elevation_gain_m", "congestion_index", "emission"])
if df.empty:
    print("⚠️ Not enough data to train. Exiting.")
    exit()

# Encode fuel types
fuel_map = {"petrol": 0, "diesel": 1, "ev": 2, "bike": 3, "bus": 4}
df["fuel_encoded"] = df["fuel_type"].str.lower().map(fuel_map).fillna(0)

# Feature and label selection
X = df[["distance_km", "elevation_gain_m", "congestion_index", "base_emission_per_km", "fuel_encoded"]]
y = df["emission"]

from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.0)
X_reduced = selector.fit_transform(X)

# Preserve only non-constant features
kept_features = X.columns[selector.get_support(indices=True)]
X = pd.DataFrame(X_reduced, columns=kept_features)

# ===============================
# 🧠 2. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# 🌲 3. Random Forest Model
# ===============================
rf_model = RandomForestRegressor(
    n_estimators=250,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_preds)
rf_mae = mean_absolute_error(y_test, rf_preds)

print(f"🌲 Random Forest → Accuracy: {rf_r2:.3f}, MAE: {rf_mae:.3f}")

# ===============================
# 💡 4. LightGBM Model
# ===============================
lgb_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
lgb_model.fit(X_train, y_train)

lgb_preds = lgb_model.predict(X_test)
lgb_r2 = r2_score(y_test, lgb_preds)
lgb_mae = mean_absolute_error(y_test, lgb_preds)

print(f"💡 LightGBM → Accuracy: {lgb_r2:.3f}, MAE: {lgb_mae:.3f}")

# ===============================
# 💾 5. Save Models
# ===============================
joblib.dump(rf_model, "emission_rf.pkl")
joblib.dump(lgb_model, "emission_lgbm.pkl")

print("✅ Both models saved successfully!")

# ===============================
# 🧾 6. Optional: Auto-update app.py MODEL_SCORES
# ===============================
try:
    if os.path.exists("app.py"):
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()

        # Update MODEL_SCORES if present
        import re
        pattern = r'MODEL_SCORES\s*=\s*\{[^}]*\}'
        new_scores = f'MODEL_SCORES = {{"Random Forest": {rf_r2:.2f}, "LightGBM": {lgb_r2:.2f}}}'
        content = re.sub(pattern, new_scores, content)

        with open("app.py", "w", encoding="utf-8") as f:
            f.write(content)

        print("🔄 Updated MODEL_SCORES in app.py automatically.")
except Exception as e:
    print(f"⚠️ Could not update app.py: {e}")

# ===============================
# 🌍 7. Summary
# ===============================
print("\n📊 Training Summary:")
print(f"Random Forest → Accuracy = {rf_r2:.3f}, MAE = {rf_mae:.3f}")
print(f"LightGBM      → Accuracy = {lgb_r2:.3f}, MAE = {lgb_mae:.3f}")
print("\n✅ Models ready for use in GreenRoute 🌿")
