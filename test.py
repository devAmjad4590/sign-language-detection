import numpy as np
import pandas as pd
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# **Loading and Preprocessing the Dataset**

# Load the dataset from .npz file
data = np.load('asl_dataset.npz')

# Extract training, validation, and test datasets
train_images = data['train_images']
train_labels = data['train_labels']
val_images = data['val_images']
val_labels = data['val_labels']
test_images = data['test_images']
test_labels = data['test_labels']
class_names = data['class_names']

# **Resize images to 64x64**
train_images = np.array([cv2.resize(img, (64, 64)) for img in train_images])
val_images = np.array([cv2.resize(img, (64, 64)) for img in val_images])
test_images = np.array([cv2.resize(img, (64, 64)) for img in test_images])

# **Add an additional dimension to the image arrays**
train_images = np.expand_dims(train_images, axis=-1)
val_images = np.expand_dims(val_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# **Convert Labels to Binary Format**
lb = LabelBinarizer()
y_train = lb.fit_transform(train_labels)
y_val = lb.transform(val_labels)
y_test = lb.transform(test_labels)

# **Determine the Number of Classes**
num_classes = y_train.shape[1]

# **Image Augmentation and Normalization**
train_datagen = ImageDataGenerator(
    rescale=1./255,
    height_shift_range=0.2,
    width_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# **Data Visualization**
# Plot the first 10 images from the dataset
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for i in range(10):
    axes[i].imshow(train_images[i].reshape(64, 64), cmap='gray')
    axes[i].set_title(f'Label: {class_names[train_labels[i]]}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# **Building the CNN Model**
model = Sequential()

# Add convolutional and pooling layers
model.add(Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu', input_shape=(64, 64, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))

model.add(Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))

model.add(Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))

model.add(Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))

# Flatten and add dense layers
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=num_classes, activation='softmax'))  # Update the number of units to match the number of classes

# Display model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# **Training the Model**
history = model.fit(
    train_datagen.flow(train_images, y_train, batch_size=32),
    epochs=20,
    validation_data=(val_images, y_val),
    shuffle=True
)

# **Save the trained model**
model.save('asl_cnn_model.h5')

# **Evaluating the Model**
loss, accuracy = model.evaluate(test_images, y_test)
print(f'MODEL ACCURACY = {accuracy * 100:.2f}%')

# You can plot the accuracy/loss if you want to further visualize the training process