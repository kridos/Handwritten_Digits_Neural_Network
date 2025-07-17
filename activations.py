import numpy as np

def relu(x):
    return np.maximum(0, x)  

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    norm_base = np.sum(exp, axis=1, keepdims=True) #gives us the sum of rows
    return exp / norm_base
    
