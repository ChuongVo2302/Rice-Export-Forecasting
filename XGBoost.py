import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv('rice_data.csv')

# 2. Feature engineering for tabular/tree-based models
# Build dates from row order (continuous monthly data, Jan 2005 - Dec 2022).
# The Month column in rice_data.csv was mangled by a spreadsheet, so it is not parsed.
df['Date'] = pd.date_range(start='2005-01-01', periods=len(df), freq='MS')
df['Month_Num'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['Year'] = df['Date'].dt.year

# Create lag features for volume and key drivers
for lag in [1, 2, 3, 6, 12]:
    df[f'Volume_Lag_{lag}'] = df['Vietnam Exporting Volume (Tons)'].shift(lag)

# Rolling statistics
df['Volume_MA_3'] = df['Vietnam Exporting Volume (Tons)'].shift(1).rolling(window=3).mean()
df['Volume_MA_6'] = df['Vietnam Exporting Volume (Tons)'].shift(1).rolling(window=6).mean()

# Drop rows with NaN from lagging
df_model = df.dropna().reset_index(drop=True)

feature_cols = [
    'Month_Num', 'Quarter', 'Year',
    'Exchange Rate (thousand VND/Dollar)',
    'Hour of Sunlight (Hour)', 'Average Temperature (Celcius)',
    'Average Moisture ( )', 'Average Rainfall (mm)',
    'Thailand (Value)', 'Thailand (Volume)',
    'Volume_Lag_1', 'Volume_Lag_2', 'Volume_Lag_3', 'Volume_Lag_6', 'Volume_Lag_12',
    'Volume_MA_3', 'Volume_MA_6'
]
target_col = 'Vietnam Exporting Volume (Tons)'

X = df_model[feature_cols]
y = df_model[target_col]

# Chronological split (80% train, 20% test)
train_size = int(len(df_model) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Train XGBoost Regressor
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"XGBoost Test RMSE: {rmse:.2f}")
print(f"XGBoost Test MAE: {mae:.2f}")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test)), y_test.values, label='Actual Export Volume', color='blue', marker='o')
plt.plot(range(len(y_pred)), y_pred, label='XGBoost Predicted Volume', color='green', linestyle='--', marker='^')
plt.title('XGBoost: Vietnam Rice Export Volume Forecasting')
plt.xlabel('Months in Test Set')
plt.ylabel('Export Volume (Tons)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Results/xgboost_rice_export_results.png')
plt.close()

# Feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 5 Important Features:")
print(importances.head(5))