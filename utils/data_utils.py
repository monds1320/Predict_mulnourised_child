import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

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
        rescale=1.0 / 255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator, test_generator


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


def predict_from_generator(model, data_generator):
    """
    Perform predictions on a dataset using a given model.

    Parameters:
    - model (tf.keras.Model): Pre-trained Keras model.
    - data_generator (DirectoryIterator): Image data generator.

    Returns:
    - predictions (np.ndarray): Array of predictions.
    """
    predictions = model.predict(data_generator)
    return predictions


def count_images(directory):
    """
    Counts the number of images in a directory.

    Parameters:
    - directory (str): Path to the directory containing images.

    Returns:
    - count (int): Number of images found in the directory.
    """
    count = sum([len(files) for r, d, files in os.walk(directory)])
    return count


def main():
    # Example usage of functions in this module
    train_dir = 'data/train'
    test_dir = 'data/test'
    model_path = 'models/cnn_model.keras'  # Adjust this to your model's actual path

    train_count = count_images(train_dir)
    test_count = count_images(test_dir)

    print(f"Number of training images: {train_count}")
    print(f"Number of testing images: {test_count}")

    train_generator, test_generator = create_data_generators(train_dir, test_dir)
    print("Data generators created successfully.")

    model = load_model(model_path)
    if model is not None:
        train_predictions = predict_from_generator(model, train_generator)
        test_predictions = predict_from_generator(model, test_generator)

        print(f"Train predictions shape: {train_predictions.shape}")
        print(f"Test predictions shape: {test_predictions.shape}")


if __name__ == '__main__':
    main()
