import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def load_model(model_path):
    """
    Load a pre-trained Keras model from a specified path.

    Parameters:
    - model_path (str): Path to the Keras model file (.h5 or .keras format).

    Returns:
    - model (tf.keras.Model): Loaded Keras model.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_data_generators(train_dir, test_dir, target_size=(150, 150), batch_size=32):
    """
    Creates train and test data generators for image data.

    Parameters:
    - train_dir (str): Path to the training data directory.
    - test_dir (str): Path to the testing data directory.
    - target_size (tuple): Tuple specifying the dimensions to which all images found will be resized.
    - batch_size (int): Size of the batches of data.

    Returns:
    - train_generator (DirectoryIterator): Iterator for training data.
    - test_generator (DirectoryIterator): Iterator for testing data.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False  # Important for getting true labels in order
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False  # Important for getting true labels in order
    )

    return train_generator, test_generator


def plot_training_history(history):
    """
    Plot training history (accuracy and loss) of a Keras model.

    Parameters:
    - history (tf.keras.callbacks.History): History object returned from model.fit().
    """
    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, test_generator):
    """
    Plot confusion matrix for the model's predictions on the test set.

    Parameters:
    - model (tf.keras.Model): Trained Keras model.
    - test_generator (DirectoryIterator): Test data generator.
    """
    # Make predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # Classification report
    print(classification_report(y_true, y_pred, target_names=test_generator.class_indices))


def main():
    train_dir = 'data/train'
    test_dir = 'data/test'
    model_path = 'models/cnn_model.keras'

    train_generator, test_generator = create_data_generators(train_dir, test_dir)
    model = load_model(model_path)

    if model is not None:
        # Plot training history
        plot_training_history(history)

        # Plot confusion matrix
        plot_confusion_matrix(model, test_generator)


if __name__ == '__main__':
    main()
