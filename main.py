import os
from predictions.predict import load_image, predict_image
import tensorflow as tf

def main():
    # Define paths
    base_dir = 'G:/malnutrition_detection_project'
    model_path = os.path.join(base_dir, 'models/cnn_model.keras')
    image_path = os.path.join(base_dir, 'check_image_02.jpeg')

    print(f"Model path: {os.path.abspath(model_path)}")
    print(f"Image path: {os.path.abspath(image_path)}")

    # Ensure the model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

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
        if prediction < 0.5:
            print("Result: Malnourished")
        else:
            print("Result: Healthy")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()