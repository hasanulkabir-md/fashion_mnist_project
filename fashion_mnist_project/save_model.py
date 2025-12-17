import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import os

# Create a simple model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model (using dummy data for illustration)
x_train = np.random.random((1000, 784))
y_train = np.random.randint(10, size=(1000,))
model.fit(x_train, y_train, epochs=5)

# Specify the path to save the model
model_path = os.path.abspath("correct/path/to/object_detection_model.keras")

# Create the directory if it does not exist
dir_path = os.path.dirname(model_path)
os.makedirs(dir_path, exist_ok=True)
print(f"Directory created or already exists: {dir_path}")

# Save the model using model.save
model.save(model_path)
print(f"Model saved at: {model_path}")
