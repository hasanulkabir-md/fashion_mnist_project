import tensorflow as tf

def create_model():
    """Define the CNN model architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 classes for Fashion MNIST
    ])
    return model

def train_model(train_images, train_labels, val_images, val_labels):
    """Train the model on the Fashion MNIST dataset."""
    model = create_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=10,
        batch_size=32
    )

    # Save the trained model
    model.save("models/fashion_mnist_model.keras")
    print("Model saved to models/fashion_mnist_model.keras")
    return history
