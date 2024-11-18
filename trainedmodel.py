# app.py

# Import necessary libraries
import os
import sys

# Print Python executable and version to verify the correct interpreter is used
print("Python executable being used:", sys.executable)
print("Python version:", sys.version)

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Set image dimensions and training parameters
IMG_HEIGHT, IMG_WIDTH = 227, 227  # For AlexNet input size
BATCH_SIZE = 32
EPOCHS = 20

# Define the directory where your images are stored
# Using os.path.join to construct the path relative to the script's location
data_dir = os.path.join(os.path.dirname(__file__), 'data')

# Initialize ImageDataGenerator with validation split and augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80% training, 20% validation
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create training data generator
train_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Create validation data generator
validation_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False  # Important for evaluation
)

# Build the AlexNet model
model = Sequential()

# Add input layer
model.add(Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)))

# Layer 1
model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=4, activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# Layer 2
model.add(Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# Layer 3
model.add(Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'))

# Layer 4
model.add(Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'))

# Layer 5
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# Flatten
model.add(Flatten())

# Fully connected layer 1
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

# Fully connected layer 2
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 9. Set up callbacks
checkpoint = ModelCheckpoint(
    filepath='best_model',
    monitor='val_accuracy',
    save_best_only=True,
    save_format='tf'  # Specify the save format explicitly
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# Calculate steps per epoch and validation steps
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint],
    verbose=2
)

# Save the final model
model.save('pneumonia_detection_model', save_format='tf')

# Plot training and validation accuracy and loss values
plt.figure(figsize=(8, 8))

# Plot accuracy
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

# Plot loss
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Binary Crossentropy')
plt.xlabel('Epoch')
plt.title('Training and Validation Loss')
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Evaluate the model on the validation data
validation_generator.reset()
val_loss, val_accuracy = model.evaluate(
    validation_generator,
    steps=validation_steps
)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')

# Generate predictions and true labels
validation_generator.reset()
Y_pred = model.predict(
    validation_generator,
    steps=validation_steps
)
y_pred = (Y_pred > 0.5).astype(int).reshape(-1)
y_true = validation_generator.classes[:len(y_pred)]  # Adjust length if necessary

# Classification report and confusion matrix
print('Classification Report')
target_names = list(validation_generator.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=target_names))

print('Confusion Matrix')
print(confusion_matrix(y_true, y_pred))
