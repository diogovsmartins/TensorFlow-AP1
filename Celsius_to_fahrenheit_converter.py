import tensorflow as tensorFlow
import numpy as np

# Sample data in Celsius
celsius_training_inputs = np.array([-40, -10, 0, 8, 15, 22, 38, 13, 25, 32], dtype=float)

# Corresponding Fahrenheit temperatures rounded to the nearest whole number (0,4=0 and 0.5 and above =1)
fahrenheit_training_inputs = np.array([-40, 14, 32, 46, 59, 72, 100, 55, 77, 90], dtype=float)

# Create a simple neural network model with 1 neuron that receives 1 entry value
model = tensorFlow.keras.Sequential([
    tensorFlow.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model specifying the loss function, the learning rate and the optimizer that will adjust
# the weights and biases to minimize the loss
model.compile(loss='mean_squared_error', optimizer=tensorFlow.keras.optimizers.Adam(0.1))

# Train the model using the fit method providing the training inputs 
# and specify how many epochs (number of training iterations), the verbose attribute just prints the results
history = model.fit(celsius_training_inputs, fahrenheit_training_inputs, epochs=1000, verbose=True)
print("Finished training the model")

# Convert Celsius to Fahrenheit using the trained model
celsius_to_convert=0.0

while celsius_to_convert != -1000:
    celsius_to_convert = float(input("\nType a celsius value, if you want to stop type -1000: "))
    fahrenheit_predicted = model.predict([celsius_to_convert])[0][0]
    print(f"{celsius_to_convert} degrees Celsius is approximately {fahrenheit_predicted:.2f} degrees Fahrenheit")

print("End of the program.")