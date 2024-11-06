
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import numpy as np
import struct
import matplotlib.pyplot as plt

# Define paths to the MNIST .idx files
train_images_path = 'data/train-images.idx3-ubyte'
train_labels_path = 'data/train-labels.idx1-ubyte'
test_images_path = 'data/t10k-images.idx3-ubyte'
test_labels_path = 'data/t10k-labels.idx1-ubyte'

# Function to load MNIST images
def load_images(file_path):
    with open(file_path, 'rb') as f:
        _, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
        return images / 255.0  # Normalize pixel values

# Function to load MNIST labels
def load_labels(file_path):
    with open(file_path, 'rb') as f:
        _, num_labels = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

# Load and preprocess the dataset
x_train = load_images(train_images_path)
y_train = load_labels(train_labels_path)
x_test = load_images(test_images_path)
y_test = load_labels(test_labels_path)

# Convert labels to categorical format
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten 28x28 images
    Dense(128, activation='relu'),  # First hidden layer
    Dense(64, activation='relu'),   # Second hidden layer
    Dense(10, activation='softmax') # Output layer for 10 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Function to test a new image
def predict_new_image(img_path):
    # Load and preprocess the custom image
    img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Reshape for model input
    
    # Display the image
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title("Input Image")
    plt.axis('off')
    plt.show()
    
    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    print(f"Predicted Class: {predicted_class[0]}")

# Test with the sample image
predict_new_image("data/sample_image.png")
