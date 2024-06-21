import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import datetime

def create_cnn_model(input_shape=(150, 150, 3)):
    base_model = VGG16(include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model layers

    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model, base_model

def create_data_generators(train_dir, test_dir, target_size=(150, 150), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

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

def train_model(train_dir, test_dir, model_save_path='models/cnn_model.keras', epochs=20):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    model, base_model = create_cnn_model()
    train_generator, test_generator = create_data_generators(train_dir, test_dir)
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, mode='min')

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=test_generator,
        validation_steps=test_generator.samples // test_generator.batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard_callback]
    )

    # Fine-tuning: Unfreeze some layers of the base model
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) // 2  # Unfreeze the last half layers

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    fine_tune_epochs = 10
    total_epochs = epochs + fine_tune_epochs

    history_fine = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=test_generator,
        validation_steps=test_generator.samples // test_generator.batch_size,
        epochs=fine_tune_epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard_callback]
    )

    return model, history, history_fine

# Example usage:
if __name__ == '__main__':
    train_dir = 'data/train'
    test_dir = 'data/test'
    model_save_path = 'G:/malnutrition_detection_project/models/cnn_model.keras'
    epochs = 20

    print("Current working directory:", os.getcwd())
    print("Model save path:", model_save_path)
    
    model, history, history_fine = train_model(train_dir, test_dir, model_save_path, epochs)
    print("Training completed and model saved to", model_save_path)

