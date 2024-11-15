# Advanced Walkthrough of Cats vs. Dogs Classification Using TensorFlow
[direct solution cellwise](DirectSolution.md)


## Table of Contents
1. Project Setup and Imports
2. Dataset Preparation and Analysis
3. Data Augmentation and Preprocessing
4. Model Architecture Definition
5. Model Compilation and Training
6. Model Evaluation and Plotting Results
7. Challenge Evaluation Logic

---

### Step 1: Project Setup and Imports

```python
try:
  # This command only in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
```

**What am i doing?**
I am importing TensorFlow and its relevant modules for creating and training a Convolutional Neural Network (CNN). The `try-except` block is for compatibility with Google Colab, ensuring TensorFlow 2.x is used.

**Why import specific modules?**
- `Sequential` and `Layers` (e.g., `Dense`, `Conv2D`, etc.) are essential for defining our CNN architecture.
- `ImageDataGenerator` is crucial for loading and augmenting our images during training.
- `os`, `np`, and `plt` are utility libraries for file handling, numerical operations, and visualization.

---

### Step 2: Download and Extract Dataset

```python
# Get project files
!wget https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip
!unzip cats_and_dogs.zip

PATH = 'cats_and_dogs'
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')
```

**What am i doing?**
I am  downloading the dataset and unzip it. The data is organized into three folders: `train`, `validation`, and `test`, containing images of cats and dogs.

**Why am i structuring the data this way?**
The split between `train`, `validation`, and `test` helps with model training, tuning, and evaluation to prevent overfitting and improve generalization.

---

### Step 3: Counting Files in Each Directory

```python
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = len(os.listdir(test_dir))
```

**What am i doing?**
i am counting the total images in each directory.

**Why count files?**
Knowing the dataset size helps us set parameters for batch processing, monitor resource usage, and ensure a balanced dataset.

---

### Step 4: Define Training Parameters

```python
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
```

**What am i setting here?**
1. **Batch size:** Determines how many images the model will process before updating weights.
2. **Epochs:** Specifies the number of complete passes through the training data.
3. **IMG_HEIGHT/IMG_WIDTH:** Defines the dimensions to resize images for model input.

**Why choose these values?**
A batch size of 128 is a good balance for memory usage and training speed. 15 epochs are typical for early tests, with image dimensions of 150x150 balancing model performance and computational efficiency.

---

### Step 5: Set Up Image Data Generators

```python
train_image_generator = None
validation_image_generator = None
test_image_generator = None
train_data_gen = None
val_data_gen = None
test_data_gen = None
```

**What’s happening here?**
I am initializing data generators for loading and augmenting training, validation, and test datasets.

**Why initialize as `None`?**
This initialization ensures the variables are defined, preparing us to assign actual data generators to them in the next steps.

---

### Step 6: Define Image Augmentation Function

```python
def plotImages(images_arr, probabilities = False):
    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5,len(images_arr) * 3))
    if probabilities is False:
      for img, ax in zip( images_arr, axes):
          ax.imshow(img)
          ax.axis('off')
    else:
      for img, probability, ax in zip(images_arr, probabilities, axes):
          ax.imshow(img)
          ax.axis('off')
          if probability > 0.5:
              ax.set_title("%.2f" % (probability*100) + "% dog")
          else:
              ax.set_title("%.2f" % ((1-probability)*100) + "% cat")
    plt.show()
```

**What is `plotImages` doing?**
This function displays images along with probability-based labels if provided. Probabilities indicate model confidence in classifying an image as a dog.

**Why visualize images and probabilities?**
Visualization helps ensure the augmentation works correctly and that we can verify model predictions after training.

---

### Step 7: Image Generators for Training and Validation

```python
train_image_generator = ImageDataGenerator(rescale=1./255,
                                           rotation_range=45,
                                           width_shift_range=0.15,
                                           height_shift_range=0.15,
                                           horizontal_flip=True,
                                           zoom_range=0.5)
validation_image_generator = ImageDataGenerator(rescale=1./255)
```

**What are these `ImageDataGenerators` doing?**
1. **Train generator:** Performs data augmentation, rescaling, and transforms for training images.
2. **Validation generator:** Only rescales validation images.

**Why use augmentation?**
Augmentation enhances model generalization by creating variations of each image, simulating different conditions and preventing overfitting.

---

### Step 8: Load Augmented Images for Visualization

```python
train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)
```

**What does this code do?**
1. Loads images with augmentation for the training set.
2. Displays the first five augmented images using the `plotImages` function.

**Why visualize augmented images?**
Visual inspection ensures the augmentation is applied correctly, simulating various orientations and scales.

---

### Step 9: Build CNN Model

```python
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.summary()
```

**What does this model architecture represent?**
The CNN architecture includes three Conv2D layers for feature extraction, followed by MaxPooling, flattening, and Dense layers for classification.

**Why this architecture?**
- **Conv2D layers** detect features.
- **MaxPooling** reduces spatial dimensions, improving computational efficiency.
- **Dense layers** classify extracted features, with a sigmoid output layer for binary classification (cat vs. dog).

---

### Step 10: Compile and Train the Model

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)
```

**What is happening here?**
1. We compile the model with `adam` as the optimizer, `binary_crossentropy` as the loss function, and accuracy as a metric.
2. `model.fit` trains the model on augmented images.

**Why choose these settings?**
The `adam` optimizer and `binary_crossentropy` loss function are commonly used for binary classification due to their efficiency and effectiveness.

---

### Step 11: Plotting Training and Validation Results

```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

**What’s happening here?**
I am extracting accuracy and loss metrics, plotting them to visualize model performance over epochs.

**Why plot these metrics?**
Visualization shows trends in learning, allowing us to spot overfitting if the validation accuracy diverges from training accuracy.

---

### Step 12: Evaluate Model Performance on Test Data

```python
correct = 0

for probability, answer in

 zip(model.predict(test_data_gen[0][0]), test_data_gen[0][1]):
    if answer == 0:
        answer = "Cat"
    else:
        answer = "Dog"
    if probability[0] > 0.5:
        prediction = "Dog"
    else:
        prediction = "Cat"
    if prediction == answer:
        correct += 1
print("Performance on Test data: %.2f" % (correct/len(test_data_gen[0][1]) * 100) + "%")
```

**What’s happening here?**
i am comparing model predictions to test labels, counting correct classifications.

**Why manually calculate accuracy?**
Manual calculation verifies the model’s test performance, critical for assessing real-world applicability.
