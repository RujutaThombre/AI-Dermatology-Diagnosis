import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import random

# Define the path to your dataset
dataset_path = "C:\\Users\\HP\\Desktop\\sihcode\\skin\\train"

# Define image dimensions
img_width, img_height = 224, 224

# Fraction of images to sample from each class
sample_fraction = 0.2  # Adjust as needed

# Load and preprocess images
def load_images(dataset_path, img_width, img_height, sample_fraction):
    images = []
    labels = []

    # Define the class names based on folder names
    class_names = os.listdir(dataset_path)

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(dataset_path, class_name)
        image_files = os.listdir(class_folder)
        
        # Randomly sample a fraction of images
        sample_size = int(sample_fraction * len(image_files))
        sampled_images = random.sample(image_files, sample_size)
        
        for image_filename in sampled_images:
            image_path = os.path.join(class_folder, image_filename)

            # Try to load the image; skip unreadable images
            try:
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_width, img_height))
                img = img / 255.0
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Skipping image {image_path}: {str(e)}")

    return np.array(images), np.array(labels)

# Load and preprocess sampled images
images, labels = load_images(dataset_path, img_width, img_height, sample_fraction)

# Define class names based on folder names
class_names = os.listdir(dataset_path)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create a data generator with data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Use VGG16 as a pre-trained model for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(class_names), activation='softmax')(x)

# Combine the base model and custom layers into a new model
combined_model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
combined_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Set up callbacks for model checkpoint and early stopping
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with data augmentation
combined_model.fit(
    train_datagen.flow(x_train, y_train, batch_size=32),
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint, early_stopping]
)

# Evaluate the model
loss, accuracy = combined_model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
