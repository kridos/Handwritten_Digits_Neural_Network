import numpy as np
from activations import sigmoid, sigmoid_deriv, relu, relu_deriv, softmax
from loss_functions import cross_entropy
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

np.random.seed(0)

class Layer:
    def __init__(self, inputs, neurons, activation=relu):
        self.weights = 0.1 * np.random.randn(inputs, neurons) #need to be this way for matrix multiplication to work properly
        self.biases = np.zeros((1, neurons))
        self.activation = activation
        
    def get_weights(self):
        return self.weights
    
    def forward(self, input):
        x = np.dot(input, self.weights) + self.biases
        self.output = self.activation(x)
    
        
        
class MultilayerPerceptron:
    def __init__(self):
        self.layers = []
         
    def add_layer(self, input, neurons, activation):
        self.layers.append(Layer(input, neurons, activation))
        

X, y = spiral_data(samples=100, classes=3)

layer1 = Layer(2, 3)
layer1.forward(X)

layer2 = Layer(3, 3, softmax)
layer2.forward(layer1.output)
#print(layer2.output[:5])

loss = cross_entropy(layer2.output, y)
print("Loss:", loss)

    