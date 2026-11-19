# Rice Export Forecasting

Forecasting Vietnam's monthly rice export volume from economic, climate, and competitor data. The project compares three models side by side:

- **LSTM** (PyTorch): multivariate recurrent network
- **GRU** (PyTorch): multivariate recurrent network with a lighter gated cell
- **XGBoost**: gradient-boosted trees on engineered lag and rolling features

Each script trains its model, prints test-set RMSE and MAE, and saves an *Actual vs Predicted* chart to [Results/](Results/).

## Dataset

Monthly data ending in **December 2022**, provided in two files:

| File | Used by | Coverage |
|---|---|---|
| [data.csv](data.csv) | `GRU.py`, `XGBoost.py` | Aug 2008 – Dec 2022 (164 rows), `Month` formatted as `Aug-08` |
| [rice_data.csv](rice_data.csv) | `LSTM.py` | Jan 2005 – Dec 2022 (207 rows), a longer version of the same series |

Columns:

| Column | Description |
|---|---|
| `Month` | Observation month |
| `Vietnam Exporting Volume (Tons)` | **Target**: Vietnam's rice export volume |
| `Vietnam Rice export Value (Mill. USD)` | Export value |
| `Exchange Rate (thousand VND/Dollar)` | VND/USD exchange rate |
| `Hour of Sunlight (Hour)` | Monthly sunlight hours |
| `Average Temperature (Celcius)` | Average temperature |
| `Average Moisture ( )` | Average humidity |
| `Average Rainfall (mm)` | Average rainfall |
| `Thailand (Value)` | Thailand's rice export value (main competitor) |
| `Thailand (Volume)` | Thailand's rice export volume |

## Models

### LSTM and GRU ([LSTM.py](LSTM.py), [GRU.py](GRU.py))

Both recurrent models use the same pipeline:

- **Features (5):** export volume, exchange rate, average temperature, average rainfall, Thailand's export volume
- **Scaling:** `MinMaxScaler` on all features. A separate scaler on the target converts predictions back to tons.
- **Sequences:** a sliding window of **6 months** predicts the next month's volume
- **Split:** 80% train / 20% test, kept in time order (no shuffling)
- **Architecture:** 2 stacked recurrent layers (hidden size 32) followed by a linear output layer
- **Training:** Adam (lr = 0.01), MSE loss, 150 full-batch epochs

### XGBoost ([XGBoost.py](XGBoost.py))

Tree models can't read sequences directly, so time information is turned into tabular features:

- **Calendar features:** month number, quarter, year
- **Lag features:** export volume lagged by 1, 2, 3, 6 and 12 months
- **Rolling means:** 3-month and 6-month moving averages of past volume
- **Exogenous features:** exchange rate, sunlight, temperature, humidity, rainfall, Thailand's value and volume
- **Model:** `XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)`
- **Split:** 80% train / 20% test in time order

The script also prints the 5 most important features.

## Results

Charts compare actual and predicted export volume over the test months:

<p align="center">
  <img src="Results/lstm_multivariate_results.png" alt="LSTM: actual vs predicted" width="800">
  <br><em>Multivariate LSTM</em>
</p>

<p align="center">
  <img src="Results/gru_multivariate_results.png" alt="GRU: actual vs predicted" width="800">
  <br><em>Multivariate GRU</em>
</p>

<p align="center">
  <img src="Results/xgboost_rice_export_results.png" alt="XGBoost: actual vs predicted" width="800">
  <br><em>XGBoost</em>
</p>

| Model | Test RMSE (tons) | Test MAE (tons) |
|---|---|---|
| LSTM | _fill in_ | _fill in_ |
| GRU | _fill in_ | _fill in_ |
| XGBoost | _fill in_ | _fill in_ |

## Getting Started

### Requirements

- Python 3.9+
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `torch`, `xgboost`

```bash
pip install pandas numpy scikit-learn matplotlib torch xgboost
```

### Run

Run each script from the project root so the relative paths to the CSV files and `Results/` resolve:

```bash
python LSTM.py
python GRU.py
python XGBoost.py
```

Each script prints its RMSE and MAE and writes its chart to `Results/`. `LSTM.py` and `GRU.py` also open the chart in a window.

## Project Structure

```
Rice-Export-Forecasting/
├── LSTM.py           # Multivariate LSTM (PyTorch)
├── GRU.py            # Multivariate GRU (PyTorch)
├── XGBoost.py        # XGBoost with lag/rolling features
├── data.csv          # Monthly dataset (used by GRU and XGBoost)
├── rice_data.csv     # Monthly dataset (used by LSTM)
└── Results/          # Saved prediction charts
```

## Notes

- No random seed is set for PyTorch, so LSTM and GRU results vary slightly from run to run. Add `torch.manual_seed(42)` at the top of each script for reproducible numbers.
- The scalers are fit on the full dataset before the train/test split, so a little information from the test period leaks into training. Fitting them on the training portion only would give a stricter evaluation.
