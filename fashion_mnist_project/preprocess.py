import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    # Load Fashion MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize pixel values to [0, 1]
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Split training data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42
    )

    # Add a channel dimension for CNNs
    train_images = train_images[..., tf.newaxis]
    val_images = val_images[..., tf.newaxis]
    test_images = test_images[..., tf.newaxis]

    return train_images, val_images, test_images, train_labels, val_labels, test_labels
