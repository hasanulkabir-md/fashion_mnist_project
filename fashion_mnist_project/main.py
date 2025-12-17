import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs and warnings

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress TensorFlow warnings

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

from preprocess import load_and_preprocess_data
from train import train_model
from evaluate import evaluate_model  # Import the evaluation function
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf  # Import TensorFlow

def plot_image(i, predictions_array, true_label, img):
    """Plot a single image with its predicted and true labels."""
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f"{predicted_label} ({true_label})", color=color)

def plot_value_array(i, predictions_array, true_label):
    """Plot a bar chart of prediction probabilities for a single image."""
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")  # Gray bars
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')  # Red bar for predicted label
    thisplot[true_label].set_color('blue')  # Blue bar for true label

if __name__ == "__main__":
    # Load and preprocess the data
    train_images, val_images, test_images, train_labels, val_labels, test_labels = load_and_preprocess_data()

    # Train the model
    print("Training the model...")
    history = train_model(train_images, train_labels, val_images, val_labels)

    # Evaluate the model
    print("Evaluating the model on test data...")
    evaluate_model(test_images, test_labels)  # Call the evaluation function

    # Visualize training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Save the training history plot to a file
    plt.savefig('results/training_plot.png')
    plt.show()

    # Visualize model predictions
    print("Visualizing model predictions on test images...")
    model = tf.keras.models.load_model("models/fashion_mnist_model.keras")
    predictions = model.predict(test_images)

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)
    
    # Save the model predictions plot to a file
    plt.savefig('results/model_predictions.png')
    plt.tight_layout()
    plt.show()

    print("Training completed.")

