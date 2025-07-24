import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

X = pd.read_csv("../data/processed/X_processed.csv")
y = pd.read_csv("../data/processed/y.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"متوسط الخطأ التربيعي (MSE): {mse:.2f}")
print(f"جذر متوسط الخطأ التربيعي (RMSE): {rmse:.2f}")
print(f"معامل التحديد (R-squared): {r2:.2f}")

joblib.dump(model, "../models/xgboost_model.pkl")
print("تم حفظ نموذج XGBoost في: xgboost_model.pkl")

