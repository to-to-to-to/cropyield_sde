import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes) + 1

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, self.num_layers + 1):
            self.weights.append(np.random.randn(sizes[i], sizes[i - 1]) * np.sqrt(2 / sizes[i - 1]))
            self.biases.append(np.zeros((sizes[i], 1)))

    def forward(self, X):
        self.activations = [X]
        self.z = []
        for i in range(self.num_layers):
            z = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            self.z.append(z)
            if i < self.num_layers - 1:
                a = np.tanh(z)  # Tanh activation for hidden layers
            else:
                a = z  # Linear activation for output
            self.activations.append(a)
        return self.activations[-1]

    def backward(self, X, y):
        m = X.shape[1]
        gradients = []
        dZ = self.activations[-1] - y
        for i in range(self.num_layers - 1, -1, -1):
            dW = (1 / m) * np.dot(dZ, self.activations[i].T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            gradients.append((dW, db))
            if i > 0:
                dA = np.dot(self.weights[i].T, dZ)
                dZ = dA * (1 - np.tanh(self.z[i - 1]) ** 2)  # Derivative of tanh
        return gradients[::-1]

    def update_parameters(self, gradients, learning_rate):
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][1]

# Load dataset
df = pd.read_csv("/Users/vitorihaldijiran/Desktop/cropyield_sde/artifact/datasets/CropSDEData/METEO_DEKADS_NUTS2_NL.csv")
df = df.dropna(subset=['TAVG', 'VPRES', 'WSPD', 'RELH', 'PREC'])

# Use estimated parameters from fBM (manually set)
mu_hat = 0.0025
sigma_hat = 0.15
H_hat = 0.55

# Compute log-returns of TAVG
df['log_TAVG'] = np.log(df['TAVG'])
df['returns'] = df['log_TAVG'].diff().dropna()

# Compute fBM variance process using Hurst exponent
df['fbm_variance'] = np.nan
df['fbm_variance'].iloc[0] = df['returns'].var()
np.random.seed(42)

for t in range(1, len(df)):
    vt = max(df['fbm_variance'].iloc[t-1], 1e-6)  # Ensure non-negative variance
    epsilon = np.random.normal(0, 1)
    
    # Simulate fractional Brownian motion variance
    vt_new = vt + mu_hat + sigma_hat * (abs(t+1)**(2*H_hat) - abs(t)**(2*H_hat)) * epsilon
    df.loc[df.index[t], 'fbm_variance'] = max(vt_new, 1e-6)  # Ensure non-negative variance

# Setup the columns for training and target features
X = df[['VPRES', 'WSPD', 'RELH', 'fbm_variance']].values
y = df['TAVG'].values.reshape(-1, 1)

# Split the data into training and validation datasets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the input data
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
X_train = (X_train - X_train_mean) / X_train_std
X_val = (X_val - X_train_mean) / X_train_std

# Define the MLP model
input_size = X_train.shape[1]
hidden_sizes = [128, 64, 32]  # Increased hidden layers
output_size = y_train.shape[1]
mlp = MLP(input_size, hidden_sizes, output_size)

# Training parameters
num_epochs = 5000
learning_rate = 0.005
min_learning_rate = 0.0001
learning_rate_decay = 0.99

best_loss = float("inf")
early_stopping_threshold = 100
patience = 0

# Training loop with adaptive learning rate
for epoch in range(num_epochs):
    outputs = mlp.forward(X_train.T)
    gradients = mlp.backward(X_train.T, y_train.T)
    mlp.update_parameters(gradients, learning_rate)
    
    loss = np.mean((outputs - y_train.T) ** 2)
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1} - Loss: {loss}")

    # Early stopping
    if loss < best_loss:
        best_loss = loss
        patience = 0
    else:
        patience += 1

    if patience > early_stopping_threshold:
        print(f"Early stopping at epoch {epoch+1}")
        break

    # Decay learning rate
    learning_rate = max(min_learning_rate, learning_rate * learning_rate_decay)

# Testing
test_outputs = mlp.forward(X_val.T)
test_loss = np.mean((test_outputs - y_val.T) ** 2)

# Final Metrics
print("Final Test MSE:", test_loss)
