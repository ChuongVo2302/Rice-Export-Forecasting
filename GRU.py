import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv('data.csv')
features = [
    'Vietnam Exporting Volume (Tons)', 
    'Exchange Rate (thousand VND/Dollar)', 
    'Average Temperature (Celcius)',
    'Average Rainfall (mm)',
    'Thailand (Volume)'
]
data = df[features].values

# 2. Scale Features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

scaler_target = MinMaxScaler(feature_range=(0, 1))
scaler_target.fit(data[:, 0].reshape(-1, 1))

# 3. Create Sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        xs.append(data[i:(i+seq_length)])
        ys.append(data[i+seq_length, 0])
    return np.array(xs), np.array(ys)

seq_length = 6 
X, y = create_sequences(scaled_data, seq_length)

# Chronological Split
train_size = int(len(X) * 0.8)
X_train = torch.tensor(X[:train_size], dtype=torch.float32)
y_train = torch.tensor(y[:train_size], dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X[train_size:], dtype=torch.float32)
y_test = torch.tensor(y[train_size:], dtype=torch.float32).view(-1, 1)

# 4. Define GRU Architecture
class MultivariateGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultivariateGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

model = MultivariateGRU(input_size=len(features), hidden_size=32, num_layers=2, output_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 5. Train the Model
epochs = 150
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()

# 6. Evaluate and Inverse Transform
model.eval()
test_predictions = scaler_target.inverse_transform(model(X_test).detach().numpy())
actual_values = scaler_target.inverse_transform(y_test.detach().numpy())

# Calculate metrics
rmse = np.sqrt(mean_squared_error(actual_values, test_predictions))
mae = mean_absolute_error(actual_values, test_predictions)
print(f"GRU Test RMSE: {rmse:.2f}")
print(f"GRU Test MAE: {mae:.2f}")

# 7. Plotting
plt.figure(figsize=(10, 5))
plt.plot(actual_values, label='Actual Export Volume', color='blue', marker='o')
plt.plot(test_predictions, label='GRU Predicted Volume', color='purple', linestyle='--', marker='s')

plt.title('Multivariate GRU: Vietnam Rice Export Volume Prediction')
plt.xlabel('Months in Test Set')
plt.ylabel('Export Volume (Tons)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Results/gru_multivariate_results.png')
plt.show()