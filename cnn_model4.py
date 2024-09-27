import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix

# Load train and test datasets
train_data = pd.read_csv('sign_mnist_train.csv')
test_data = pd.read_csv('sign_mnist_test.csv')

y_train = train_data['label']
y_test = test_data['label']
del train_data['label']
del test_data['label']

label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_test)  # Use transform instead of fit_transform

def preprocess_image(x):
    x = x / 255.0
    x = x.reshape(-1, 28, 28, 1)  # Converting it into 28x28 grayscale image
    return x

train_x = preprocess_image(train_data.values)
test_x = preprocess_image(test_data.values)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False  # randomly flip images
)

datagen.fit(train_x)

# Define the model
model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
    Conv2D(256, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(24, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define the learning rate reduction callback
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# Train the model
history = model.fit(datagen.flow(train_x, y_train, batch_size=128),
                    epochs=5,
                    validation_data=(test_x, y_test),
                    callbacks=[lr_reduction])

# Evaluate the model
test_loss, test_acc = model.evaluate(test_x, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Save the model
model.save('asl_cnn_model.h5')