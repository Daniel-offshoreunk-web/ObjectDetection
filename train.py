import numpy as np

# load
data = np.loadtxt("training_data.txt", delimiter=",")
X = data[:, :2]  # First two columns: Circularity, Aspect Ratio
y = data[:, 2:]  # Last column: Label (yes/no)

# basic functions
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return x * (1 - x)

# Make up the weights
np.random.seed(1)
weights1 = np.random.uniform(-1, 1, (2, 4)) # 2 inputs -> 4 hidden nodes
bias1 = np.zeros((1, 4))
weights2 = np.random.uniform(-1, 1, (4, 1)) # 4 hidden nodes -> 1 output
bias2 = np.zeros((1, 1))

learning_rate = 0.1

print("Training...")
# rep 10k times
for epoch in range(10000):
    # guess
    hidden_layer = sigmoid(np.dot(X, weights1) + bias1)
    output_layer = sigmoid(np.dot(hidden_layer, weights2) + bias2)
    
    # how wrong
    error = y - output_layer
    d_output = error * sigmoid_derivative(output_layer)
    
    error_hidden = d_output.dot(weights2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer)
    
    # learn
    weights2 += hidden_layer.T.dot(d_output) * learning_rate
    bias2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights1 += X.T.dot(d_hidden) * learning_rate
    bias1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

print("\nTraining complete!")
print(f"weights1 = np.array({repr(weights1.tolist())})")
print(f"bias1 = np.array({repr(bias1.tolist()[0])})")
print(f"weights2 = np.array({repr(weights2.tolist())})")
print(f"bias2 = np.array({repr(bias2.tolist()[0])})")