#🔍 Project: Image Classification with CNN (TensorFlow)
#📦 Dependencies

pip install tensorflow matplotlib numpy

#✅ Python Script (TensorFlow-based CNN for CIFAR-10 Dataset)

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. Load and preprocess the data
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0, 1]
    return x_train, y_train, x_test, y_test

# 2. Build the CNN model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# 3. Compile and train the model
def train_model(model, x_train, y_train, x_test, y_test):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10,
                        validation_data=(x_test, y_test))
    return history

# 4. Visualize training performance
def plot_performance(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.show()

# 5. Visualize sample predictions
def show_predictions(model, x_test, y_test, class_names):
    predictions = model.predict(x_test[:9])
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(x_test[i])
        plt.title(f"Pred: {class_names[np.argmax(predictions[i])]} \nTrue: {class_names[y_test[i][0]]}")
        plt.axis('off')
    plt.show()

# Main runner
if __name__ == "__main__":
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    x_train, y_train, x_test, y_test = load_data()
    model = build_model()
    history = train_model(model, x_train, y_train, x_test, y_test)
    plot_performance(history)
    show_predictions(model, x_test, y_test, class_names)
