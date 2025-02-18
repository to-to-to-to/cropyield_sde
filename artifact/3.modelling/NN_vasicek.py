import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.tsa.stattools import adfuller

# Load dataset
data = pd.read_csv('/Users/vitorihaldijiran/Desktop/cropyield_sde/artifact/datasets/CropSDEData/METEO_DEKADS_NUTS2_NL.csv')

# Feature Selection
features = ['TAVG', 'VPRES', 'WSPD']
target = 'PREC'

# Drop missing values
data = data.dropna(subset=features + [target])

# Prepare data
X = data[features]
y = data[target]

# Ensure stationarity of target variable
if adfuller(y)[1] > 0.05:
    print("Target variable is non-stationary. Applying differencing...")
    y = y.diff().dropna()
    X = X.iloc[1:]

# Align X and y
X, y = X.iloc[:len(y)], y.iloc[:len(X)]

# Feature Engineering: Add Interaction Terms
X['TAVG_VPRES'] = X['TAVG'] * X['VPRES']

# Scale Features and Target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()



# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# ---- Optimized Vasicek MLE ----
def vasicek_mle(params, data):
    a, b, sigma = params
    dt = 1
    X = data
    residuals = X[1:] - (X[:-1] + a * (b - X[:-1]) * dt)
    log_likelihood = -np.sum(0.5 * np.log(2 * np.pi * (sigma**2 + 1e-6) * dt) + (residuals**2 / (2 * (sigma**2 + 1e-6) * dt)))
    return -log_likelihood

# Initial guesses and bounds for MLE
initial_guess = [0.1, np.mean(y), np.std(y)]
bounds = [(1e-5, 1.0), (None, None), (1e-5, 5.0)]  # Prevent extreme sigma values

# Perform MLE for Vasicek Model
res_mle = minimize(vasicek_mle, initial_guess, args=(y.values,), method='L-BFGS-B', bounds=bounds)

# Extract estimated parameters
a_mle, b_mle, sigma_mle = res_mle.x

print("\nEstimated Vasicek Parameters using Maximum Likelihood Estimation (MLE):")
print(f"Alpha (a): {a_mle}, Beta (b): {b_mle}, Sigma: {sigma_mle}")

# ---- Optimized Minimal Neural Network ----
class VasicekNN(nn.Module):
    def __init__(self, input_size):
        super(VasicekNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.activation = nn.GELU()
        self.layernorm1 = nn.LayerNorm(32)
        self.layernorm2 = nn.LayerNorm(16)

    def forward(self, x):
        x = self.activation(self.layernorm1(self.fc1(x)))
        x = self.activation(self.layernorm2(self.fc2(x)))
        x = self.fc3(x)
        return x

# Initialize the Neural Network
model = VasicekNN(X_train.shape[1])

# Optimizer & Scheduler
criterion = nn.SmoothL1Loss()  # Log-Cosh Approximation
optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5, amsgrad=True)
scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=0.01, step_size_up=100, mode='triangular2')

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ---- Efficient Training Loop ----
epochs = 3000
batch_size = 512
accumulation_steps = 4

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size(0))
    epoch_loss = 0

    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_X)

        # Log-Cosh Approximation Loss
        loss = nn.MSELoss()
        output = loss(outputs,  batch_y)
        output.backward()


        if (i // batch_size + 1) % accumulation_steps == 0 or i + batch_size >= X_train_tensor.size(0):
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
# 
        epoch_loss += output.item()
    
    scheduler.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")

# ---- Evaluate Model ----
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor).numpy()

test_mse = mean_squared_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\nNeural Network Test MSE: {test_mse}")
print(f"Neural Network Test R^2 Score: {test_r2}")

# Compare with Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)

print(f"\nLinear Regression Test MSE: {lr_mse}")
print(f"Linear Regression Test R^2 Score: {lr_r2}")

if test_mse < lr_mse:
    print("\nNeural Network outperforms Linear Regression.")
else:
    print("\nLinear Regression outperforms Neural Network.")
