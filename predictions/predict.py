import os
import numpy as np
import tensorflow as tf
from PIL import Image

def load_image(img_path, target_size=(150, 150)):
    print(f"Attempting to load image from: {img_path}")
    try:
        with Image.open(img_path) as img:
            img = img.resize(target_size)
            img = img.convert('RGB')
            img_array = np.array(img) / 255.0
            print(f"Image shape after preprocessing: {img_array.shape}")  # Debug print
            return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error loading image: {e}")
        raise

def predict_image(model, img_path):
    img_array = load_image(img_path)
    prediction = model.predict(img_array)
    print(f"Raw prediction values: {prediction}")
    return prediction[0][0]  # Return the predicted probability (or class)

def run_prediction(model_path, image_path):
    # Ensure the image file exists
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return

    # Load the trained model
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Make prediction
    try:
        prediction = predict_image(model, image_path)

        # Print prediction result
        print("Prediction:", prediction)
        if prediction > 0.5:
            print("Result: Malnourished")
        else:
            print("Result: Healthy")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == '__main__':
    model_path = 'models/cnn_model.keras'  # Path to your trained model
    image_path = 'check_new_image.jpeg'  # Path to the new image you want to predict
    
    run_prediction(model_path, image_path)
