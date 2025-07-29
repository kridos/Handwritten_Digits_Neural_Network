import numpy as np
from activations import sigmoid, sigmoid_deriv, relu, relu_deriv, softmax
from loss_functions import cross_entropy
from keras.datasets import mnist
from keras.utils import to_categorical


np.random.seed(0)

learning_rate = 1

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
        
    def backwards_output_layer(self, actual, batch_size):
        self.propogation_error = (self.output - actual) / batch_size
    
        
        
class MultilayerPerceptron:
    def __init__(self):
        self.layers = []
         
    def add_layer(self, input, neurons, activation):
        self.layers.append(Layer(input, neurons, activation))
        
(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X.reshape(-1, 28 * 28)
test_X = test_X.reshape(-1, 28 * 28)

train_y = to_categorical(train_y, num_classes=10)
test_y = to_categorical(test_y, num_classes=10)

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

layer1 = Layer(784, 10)
layer1.forward(train_X[:1])

output_layer = Layer(10, 10, softmax)
output_layer.forward(layer1.output)

print(output_layer.output)

# loss = cross_entropy(output_layer.output, y)

# output_layer.backwards_output_layer("temp placeholder", output_layer.output.length)
# print("Loss:", loss)

    