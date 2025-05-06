"""
Team 9 : Panjugula Anvita (M16590527), Pooja Putta (M16474112), Tanuj Ithesh Kumar (M16474171) 

“all team members have contributed in equal measure to this effort”,

"""


import csv
import numpy as np
import matplotlib.pyplot as plt
import os

# Making sure there is place to save our plots
os.makedirs('plots', exist_ok=True)

def load_data(filename):
    # Reading in the wine data in csv files
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)  # Skip the header
        data = [list(map(float, row)) for row in reader]
    
    # Converting to numpy array and normalize
    data = np.array(data)
    X, y = data[:, :-1], data[:, -1]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    return np.column_stack((X, y))

def split_data(data, train_pct, valid_pct):
    # Shuffling and spliting the data
    np.random.shuffle(data)
    n = len(data)
    train_size = int(n * train_pct)
    valid_size = int(n * valid_pct)
    
    train = data[:train_size]
    valid = data[train_size:train_size+valid_size]
    test = data[train_size+valid_size:]
    
    return train, valid, test

def get_X_y(data):
    return data[:, :-1], data[:, -1]

def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def predict(X, w):
    return X.dot(w)

def gradient(X, y, w):
    y_pred = predict(X, w)
    return (2 / len(y)) * X.T.dot(y_pred - y)

def batch_gd(X, y, lr, epochs, reg_type=None, reg_strength=0):
    w = np.zeros(X.shape[1])
    losses = []
    
    for _ in range(epochs):
        y_pred = predict(X, w)
        loss = mse(y, y_pred)
        losses.append(loss)
        
        grad = gradient(X, y, w)
        
        if reg_type == 'L2':
            w -= lr * (grad + 2 * reg_strength * w)
        elif reg_type == 'L1':
            w -= lr * (grad + reg_strength * np.sign(w))
        else:
            w -= lr * grad
    
    return w, losses

def mini_batch_gd(X, y, lr, epochs, batch_size, reg_type=None, reg_strength=0):
    w = np.zeros(X.shape[1])
    losses = []
    
    for _ in range(epochs):
        # Shuffling the data each in epoch
        idx = np.random.permutation(X.shape[0])
        X, y = X[idx], y[idx]
        
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            y_pred = predict(X_batch, w)
            loss = mse(y_batch, y_pred)
            losses.append(loss)
            
            grad = gradient(X_batch, y_batch, w)
            
            if reg_type == 'L2':
                w -= lr * (grad + 2 * reg_strength * w)
            elif reg_type == 'L1':
                w -= lr * (grad + reg_strength * np.sign(w))
            else:
                w -= lr * grad
    
    return w, losses

def plot_regression(X, y, w, title, filename):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.5, label='Data')
    
    # Sorting X for a smooth line plot
    idx = np.argsort(X)
    X_sorted = X[idx]
    
    X_line = np.column_stack((np.ones(X_sorted.shape), X_sorted))
    y_pred = predict(X_line, w)
    
    plt.plot(X_sorted, y_pred, color='red', linewidth=2, label='Regression line')
    plt.xlabel('Normalized Feature')
    plt.ylabel('Wine Quality')
    plt.title(title)
    plt.legend()
    plt.savefig(f'plots/{filename}.png')
    plt.close()

# getting this party started
data = load_data('wine+quality/winequality-red.csv')
train, valid, test = split_data(data, 0.7, 0.15)

X_train, y_train = get_X_y(train)
X_valid, y_valid = get_X_y(valid)
X_test, y_test = get_X_y(test)

X_train_bias = add_bias(X_train)
X_valid_bias = add_bias(X_valid)
X_test_bias = add_bias(X_test)

# Hyperparameters (feel free to change these)
lr = 0.01
epochs = 5000
batch_size = 32
reg_strength = 0.1

print("Running Batch Gradient Descent:")
w_batch, mse_batch = batch_gd(X_train_bias, y_train, lr, epochs)
print(f"MSE (no regularization): {mse_batch[-1]:.4f}")

w_batch_l2, mse_batch_l2 = batch_gd(X_train_bias, y_train, lr, epochs, 'L2', reg_strength)
print(f"MSE (L2 regularization): {mse_batch_l2[-1]:.4f}")

w_batch_l1, mse_batch_l1 = batch_gd(X_train_bias, y_train, lr, epochs, 'L1', reg_strength)
print(f"MSE (L1 regularization): {mse_batch_l1[-1]:.4f}")

print("\nRunning Mini-batch Gradient Descent:")
w_mini, mse_mini = mini_batch_gd(X_train_bias, y_train, lr, epochs, batch_size)
print(f"MSE (no regularization): {mse_mini[-1]:.4f}")

w_mini_l2, mse_mini_l2 = mini_batch_gd(X_train_bias, y_train, lr, epochs, batch_size, 'L2', reg_strength)
print(f"MSE (L2 regularization): {mse_mini_l2[-1]:.4f}")

w_mini_l1, mse_mini_l1 = mini_batch_gd(X_train_bias, y_train, lr, epochs, batch_size, 'L1', reg_strength)
print(f"MSE (L1 regularization): {mse_mini_l1[-1]:.4f}")

# Find the most important feature
best_feature = np.argmax(np.abs(w_batch[1:]))  # Ignore the bias term

# Plot all the things!
plot_regression(X_train[:, best_feature], y_train, w_batch[[0, best_feature+1]], 
                'Batch GD', 'batch_gd')
plot_regression(X_train[:, best_feature], y_train, w_batch_l2[[0, best_feature+1]], 
                'Batch GD with L2', 'batch_gd_l2')
plot_regression(X_train[:, best_feature], y_train, w_batch_l1[[0, best_feature+1]], 
                'Batch GD with L1', 'batch_gd_l1')
plot_regression(X_train[:, best_feature], y_train, w_mini[[0, best_feature+1]], 
                'Mini-batch GD', 'mini_batch_gd')
plot_regression(X_train[:, best_feature], y_train, w_mini_l2[[0, best_feature+1]], 
                'Mini-batch GD with L2', 'mini_batch_gd_l2')
plot_regression(X_train[:, best_feature], y_train, w_mini_l1[[0, best_feature+1]], 
                'Mini-batch GD with L1', 'mini_batch_gd_l1')

print("\nCheck the output plots in the 'plots' folder.")