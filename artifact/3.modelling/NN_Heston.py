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
                a = np.tanh(z)
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
                dZ = dA * (1 - np.tanh(self.z[i - 1]) ** 2)
        return gradients[::-1]

    def update_parameters(self, gradients, learning_rate):
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][1]

if __name__ == "__main__":
    df = pd.read_csv("/Users/vitorihaldijiran/Desktop/cropyield_sde/artifact/datasets/CropSDEData/METEO_DEKADS_NUTS2_NL.csv")
    df = df.dropna(subset=['TAVG', 'VPRES', 'WSPD', 'RELH', 'PREC'])
    
    # Extract Heston-estimated parameters
    mu_hat = -0.01762635440919763
    kappa_hat = 0.05360217296743761
    theta_hat = 1.1327667398172898
    sigma_hat = 0.2145933527454109
    
    # Compute log-returns and variance of TAVG
    df['log_TAVG'] = np.log(df['TAVG'])
    df['returns'] = df['log_TAVG'].diff().dropna()
    df['variance'] = df['returns'].rolling(window=30).var().dropna()
    df = df.dropna(subset=['returns', 'variance'])
    
    # Compute Heston variance series
    df['heston_variance'] = np.nan
    df['heston_variance'].iloc[0] = df['variance'].iloc[0]
    np.random.seed(42)
    for t in range(1, len(df)):
        vt = df['heston_variance'].iloc[t-1]
        epsilon = np.random.normal(0, 1)
        vt_new = vt + kappa_hat * (theta_hat - vt) + sigma_hat * np.sqrt(vt) * epsilon
        df.loc[df.index[t], 'heston_variance'] = max(vt_new, 0)
    
    # Setup the columns for training and target features
    X = df[['VPRES', 'WSPD', 'RELH', 'heston_variance']].values
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
    hidden_sizes = [64, 64]
    output_size = y_train.shape[1]
    mlp = MLP(input_size, hidden_sizes, output_size)
    
    # Training parameters
    num_epochs = 5000
    learning_rate = 0.005
    
    # Training loop
    for epoch in range(num_epochs):
        outputs = mlp.forward(X_train.T)
        gradients = mlp.backward(X_train.T, y_train.T)
        mlp.update_parameters(gradients, learning_rate)
        loss = np.mean((outputs - y_train.T) ** 2)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1} - Loss: {loss}")
        if loss < 0.05:
            break
    
    # Testing
    test_outputs = mlp.forward(X_val.T)
    test_loss = np.mean((test_outputs - y_val.T) ** 2)
    
    # Final Metrics
    print("Final Test MSE:", test_loss)
