import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs and warnings

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress TensorFlow warnings

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(test_images, test_labels):
    # Load the trained model
    model = tf.keras.models.load_model("models/fashion_mnist_model.keras")

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Predict on the test data
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate and display performance metrics
    print("Classification Report:")
    print(classification_report(test_labels, predicted_labels, target_names=[
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]))

    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, predicted_labels))

    # Visualize predictions on a few test samples
    visualize_predictions(test_images[:10], test_labels[:10], predictions[:10])

def visualize_predictions(images, labels, predictions):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"True: {class_names[labels[i]]}\nPred: {class_names[np.argmax(predictions[i])]}")
        plt.axis('off')
    plt.tight_layout()
    
    # Save the evaluated images plot to a file
    plt.savefig('results/evaluated_images.png')
    plt.show()
