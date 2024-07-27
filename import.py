import scipy.io
import numpy as np
import tensorflow as tf

# Load the .mat file
mat_contents = scipy.io.loadmat('net.mat', struct_as_record=False, squeeze_me=True)
net = mat_contents['net']

# Extract weights and biases
input_weights = net.IW[0]  # Input weights (assuming single input layer)
biases = net.b  # Biases

# Print shapes for verification
print("Input Weights shape:", input_weights.shape)
print("Biases:", biases)

# Determine the structure of the network
input_size = input_weights.shape[1]
hidden_size = input_weights.shape[0]
output_size = 1  # Assuming a single output

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_size,)),
    tf.keras.layers.Dense(units=hidden_size, activation='relu'),
    tf.keras.layers.Dense(units=output_size, activation='sigmoid')
])

# Transpose weights to match TensorFlow's expected shape
input_weights_tf = input_weights.T

# Set the weights and biases
model.layers[0].set_weights([input_weights_tf, biases[0]])
model.layers[1].set_weights([np.zeros((hidden_size, output_size)), np.array([biases[1]])])

# Verify the model
model.summary()

# Example of making predictions (replace with your actual input data)
new_data = np.random.rand(1, input_size)  # Create a random input
predictions = model.predict(new_data)
print("Predictions:", predictions)