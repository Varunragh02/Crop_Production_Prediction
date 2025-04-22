import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import sys

# Step 1: Load dataset
print("✅ Loading dataset...")
df = pd.read_csv("D:/Guvi/Crop_production_prediction/cleaned_crops_only.csv")
print("📄 Dataset shape:", df.shape)

# Step 2: Drop unnecessary columns
df.drop(columns=['Stocks', 'Laying', 'Milk Animals', 'Producing_Animals', 'Yield/Carcass Weight'], inplace=True, errors='ignore')

# Step 3: Handle missing values
imputer = SimpleImputer(strategy='mean')
df[['Area_Harvested', 'Yield_kg_per_ha', 'Production_tonnes']] = imputer.fit_transform(
    df[['Area_Harvested', 'Yield_kg_per_ha', 'Production_tonnes']]
)

# Step 4: Encode categorical variables
print("✅ Encoding categorical features...")
df_encoded = pd.get_dummies(df[['Area', 'Item']], drop_first=True, sparse=True)
print("🔣 Encoded shape:", df_encoded.shape)

# Step 5: Combine encoded features with numerical features
df_final = pd.concat([df_encoded, df[['Area_Harvested', 'Yield_kg_per_ha', 'Production_tonnes']]], axis=1)

# Step 6: Define features and target
X = df_final.drop('Production_tonnes', axis=1)
y = df_final['Production_tonnes']

# Step 7: Train/test split
print("✅ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Feature Scaling (for Lasso & Ridge)
scaler = StandardScaler(with_mean=False)  # with_mean=False for sparse matrices
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Evaluation Function
def evaluate_model(name, y_true, y_pred):
    print("📊 Model Performance:", name)
    print("R2 Score:", r2_score(y_true, y_pred))
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    sys.stdout.flush()

# Step 9: Train & Evaluate Linear Regression
print("✅ Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
evaluate_model("Linear Regression", y_test, y_pred_lr)

# Step 10: Train & Evaluate Lasso Regression
print("✅ Training Lasso Regression...")
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)
evaluate_model("Lasso Regression", y_test, y_pred_lasso)

# Step 11: Train & Evaluate Ridge Regression
print("✅ Training Ridge Regression...")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)
evaluate_model("Ridge Regression", y_test, y_pred_ridge)

# Step 12: Train & Evaluate Random Forest
print("✅ Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
evaluate_model("Random Forest", y_test, y_pred_rf)

# Step 13: Save All Models and Scaler
print("\n💾 Saving all models to a single file...")
all_models = {
    "linear_regression": lr_model,
    "lasso_regression": lasso_model,
    "ridge_regression": ridge_model,
    "random_forest": rf_model,
    "scaler": scaler
}
joblib.dump(all_models, "D:/Guvi/Crop_production_prediction/all_models.pkl")
print("✅ All models saved to all_models.pkl")
