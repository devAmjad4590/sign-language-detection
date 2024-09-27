# **American Sign Language Recognition Using CNN**

import numpy as np
import pandas as pd
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns

# **Loading and Preprocessing the Dataset**

# Load train and test datasets
train_df = pd.read_csv('sign_mnist_train.csv')
test_df = pd.read_csv('sign_mnist_test.csv')

# Separate labels from pixel data
train_label = train_df['label']
trainset = train_df.drop(['label'], axis=1)
test_label = test_df['label']
testset = test_df.drop(['label'], axis=1)

# Reshape datasets to match CNN input shape (28x28x1 for grayscale)
X_train = trainset.values.reshape(-1, 28, 28, 1)
X_test = testset.values.reshape(-1, 28, 28, 1)

# Plot the first 10 images from the dataset
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for i in range(10):
    axes[i].imshow(X_train[i].reshape(28, 28), cmap='gray')
    axes[i].set_title(f'Label: {train_label[i]}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# **Convert Labels to Binary Format**
lb = LabelBinarizer()
y_train = lb.fit_transform(train_label)
y_test = lb.fit_transform(test_label)

# **Image Augmentation and Normalization**
train_datagen = ImageDataGenerator(
    rescale=1./255,
    height_shift_range=0.05,
    width_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)
X_test = X_test / 255.0  # Normalize test data

# **Data Visualization**
# Preview a few images
fig, axe = plt.subplots(2, 2)
fig.suptitle('Preview of dataset')
axe[0, 0].imshow(X_train[0].reshape(28, 28), cmap='gray')
axe[0, 0].set_title('label: 3  letter: C')
axe[0, 1].imshow(X_train[1].reshape(28, 28), cmap='gray')
axe[0, 1].set_title('label: 6  letter: F')
axe[1, 0].imshow(X_train[2].reshape(28, 28), cmap='gray')
axe[1, 0].set_title('label: 2  letter: B')
axe[1, 1].imshow(X_train[4].reshape(28, 28), cmap='gray')
axe[1, 1].set_title('label: 13  letter: M')

# Frequency plot of labels
plt.figure()
sns.countplot(train_label)
plt.title("Frequency of each label")

# **Building the CNN Model**
model = Sequential()

# Add convolutional and pooling layers
model.add(Conv2D(128, kernel_size=(5, 5), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(3, 3), strides=2, padding='same'))
model.add(Conv2D(64, kernel_size=(2, 2), strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Conv2D(32, kernel_size=(2, 2), strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))

# Flatten and add dense layers
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=24, activation='softmax'))

# Display model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# **Training the Model**
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=200),
    epochs=35,
    validation_data=(X_test, y_test),
    shuffle=True
)

# **Save the trained model**
model.save('asl_cnn_model.h5')

# **Evaluating the Model**
loss, accuracy = model.evaluate(X_test, y_test)
print(f'MODEL ACCURACY = {accuracy * 100:.2f}%')

# You can plot the accuracy/loss if you want to further visualize the training process
