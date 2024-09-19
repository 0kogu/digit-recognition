import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data by scaling pixel values to the range of 0 to 1
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Reshape the data to match the input shape for the model
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Convert labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# Build the CNN model
model = keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),  # Input layer
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),  # Convolution layer
        layers.MaxPooling2D(pool_size=(2, 2)),  # Pooling layer
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),  # Another convolution
        layers.MaxPooling2D(pool_size=(2, 2)),  # Pooling
        layers.Flatten(),  # Flatten the output
        layers.Dense(128, activation="relu"),  # Fully connected layer
        layers.Dense(10, activation="softmax"),  # Output layer (10 digits)
    ]
)

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Display model architecture
model.summary()



# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")


# Predict on the test set
predictions = model.predict(x_test)

# Function to plot images with predicted labels
def plot_image(pred, actual, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[..., 0], cmap=plt.cm.binary)
    predicted_label = np.argmax(pred)
    true_label = np.argmax(actual)
    color = "green" if predicted_label == true_label else "red"
    plt.xlabel(f"Pred: {predicted_label} ({true_label})", color=color)

# Display some test images with predictions
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plot_image(predictions[i], y_test[i], x_test[i])
plt.show()
