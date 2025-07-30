import numpy as np
from activations import sigmoid, sigmoid_deriv, relu, relu_deriv, softmax
from loss_functions import cross_entropy
from keras.datasets import mnist
from keras.utils import to_categorical


np.random.seed(0)


#TODO: Update this class later to make the Multilayer Perceptron more adaptable        
class MultilayerPerceptron:
    def __init__(self):
        self.layers = []
         
    def add_layer(self, input, neurons, activation, activation_deriv):
        self.layers.append(Layer(input, neurons, activation, activation_deriv))
        
def batch_generator(X, y, batch_size = 32):
    n_samples = X.shape[0]
    for i in range(0, n_samples, batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]


class Layer:
    def __init__(self, inputs, neurons, activation=relu, activation_deriv=relu_deriv):
        self.learning_rate = 0.000225
        self.weights = np.random.randn(inputs, neurons) * np.sqrt(2. / inputs)
 #need to be this way for matrix multiplication to work properly
        self.biases = np.zeros((1, neurons))
        self.activation = activation
        self.activation_deriv = activation_deriv
        
    def get_weights(self):
        return self.weights
    
    def forward(self, input):
        self.input = input
        self.z = np.dot(input, self.weights) + self.biases
        self.output = self.activation(self.z)
        return self.output
        
    def backwards_output_layer(self, y):
        batch_size = y.shape[0]
        dz = (self.output - y) / batch_size
        dW = self.input.T @ dz
        db = np.sum(dz, axis=0, keepdims=True)
        
        self.weights = self.weights - (self.learning_rate * dW)
        self.biases = self.biases - (self.learning_rate * db)
        
        return dz, self.weights
    
    def backwards_hidden_layers(self, d_next, weights_next):
        batch_size = self.input.shape[0]
        
        d = (d_next @ weights_next.T) * relu_deriv(self.z)
        dW = (self.input.T @ d) / batch_size
        db = np.sum(d, axis=0, keepdims=True) / batch_size
        
        self.weights = self.weights - (self.learning_rate * dW)
        self.biases = self.biases - (self.learning_rate * db)
        
        return d, self.weights
        
        
        
(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X.reshape(-1, 28 * 28)
test_X = test_X.reshape(-1, 28 * 28)

train_y = to_categorical(train_y, num_classes=10)
test_y = to_categorical(test_y, num_classes=10)

layer1 = Layer(784, 128)

output_layer = Layer(128, 10, softmax)

epochs = 100
for epoch in range(epochs):
    epoch_loss = 0
    batch_count = 0
    
    if epochs % 10 == 0 and epochs != 0:
        layer1.learning_rate *= 0.97
        output_layer.learning_rate *= 0.97
    
    for x_batch, y_batch in batch_generator(train_X, train_y, 16):
        
        layer1_out = layer1.forward(x_batch)
        predicted_output = output_layer.forward(layer1_out)


        epoch_loss += cross_entropy(predicted_output, y_batch)
        batch_count += 1
        
        d_output, weights_output = output_layer.backwards_output_layer(y_batch)
        layer1.backwards_hidden_layers(d_output, weights_output)
    
    
    avg_loss = epoch_loss / batch_count
    print(f'Epoch {epoch + 1}, Avg Loss = {avg_loss:.4f}')
  
total_correct = 0  

for i in range(test_X.shape[0]):
    layer1_out = layer1.forward(test_X[i:i+1])
    predicted_output = output_layer.forward(layer1_out)
    
    predicted_classes = np.argmax(predicted_output, axis=1)
    true_classes = np.argmax(test_y[i:i+1], axis=1)
    
    if predicted_classes == true_classes:
        total_correct += 1
        
print(f'Accuracy: {total_correct / test_X.shape[0] * 100:.2f}%')


    
    

    
    
    
layer1_out = layer1.forward(test_X[2:3])
predicted_output = output_layer.forward(layer1_out)



    




    