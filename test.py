import numpy as np

# Load the dataset from the saved .npz file
data = np.load('asl_dataset.npz')

# Extract the training, validation, and test datasets
train_images = data['train_images']
train_labels = data['train_labels']
val_images = data['val_images']
val_labels = data['val_labels']
test_images = data['test_images']
test_labels = data['test_labels']
class_names = data['class_names']

print("Dataset loaded successfully!")
print(f"Training set size: {train_images.shape}")
print(f"Validation set size: {val_images.shape}")
print(f"Test set size: {test_images.shape}")

# Visualize some images from the training set
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
for i in range(24):
    plt.subplot(3, 8, i + 1)
    plt.imshow(train_images[i].squeeze(), cmap='gray')  # Squeeze to remove the grayscale channel dimension
    plt.title(class_names[train_labels[i]])
    plt.axis('off')
plt.show()
