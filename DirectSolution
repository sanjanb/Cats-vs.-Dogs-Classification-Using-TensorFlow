### Cell 1
The first cell imports the necessary libraries. You shouldn’t need to modify this cell; it’s setting up the environment.

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

### Cell 2
This cell downloads the dataset and sets up the project file paths and directories.

```python
!wget https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip
!unzip cats_and_dogs.zip

PATH = 'cats_and_dogs'

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = len(os.listdir(test_dir))

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
```

### Cell 3
This cell is where you start defining the data generators. Image data generators are used to rescale and preprocess images for the neural network.

```python
# 3
train_image_generator = ImageDataGenerator(rescale=1./255)  # rescale images for training
validation_image_generator = ImageDataGenerator(rescale=1./255)  # rescale images for validation
test_image_generator = ImageDataGenerator(rescale=1./255)  # rescale images for testing

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                         directory=test_dir,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode=None,
                                                         shuffle=False)
```

Explanation:
- **Rescale:** `ImageDataGenerator` converts image pixel values to between 0 and 1 by dividing by 255, which standardizes input for the neural network.
- **Flow from directory:** `flow_from_directory` loads images from folders; we specify `target_size`, `batch_size`, and `class_mode` (set to `None` for test data).

### Cell 4
This cell defines a function to plot images. You don’t need to modify this code.

```python
def plotImages(images_arr, probabilities=False):
    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5, len(images_arr) * 3))
    if probabilities is False:
        for img, ax in zip(images_arr, axes):
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

sample_training_images, _ = next(train_data_gen)
plotImages(sample_training_images[:5])
```

### Cell 5
In this cell, you will add data augmentation techniques to help prevent overfitting by generating variations of images.

```python
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

Explanation:
- **Data augmentation**: The parameters such as `rotation_range`, `zoom_range`, and `horizontal_flip` add random variations to the images to improve generalization.

### Cell 6
Now, you can generate training data with the augmented images.

```python
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)
```

### Cell 7
In this cell, you will define the convolutional neural network (CNN) model.

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
```

Explanation:
- **Conv2D and MaxPooling2D layers**: These layers extract features and reduce dimensionality.
- **Flatten and Dense layers**: Flattening prepares data for fully connected layers, and Dense layers learn complex patterns.
- **Output layer**: The `sigmoid` function outputs probabilities for binary classification (cat or dog).

### Cell 8
Train the model using `fit`.

```python
history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)
```

### Cell 9
This cell visualizes the accuracy and loss.

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

### Cell 10
Finally, use your model to predict whether the images in the test set are of cats or dogs.

```python
probabilities = model.predict(test_data_gen)
plotImages([test_data_gen[i] for i in range(50)], probabilities)
```

### Cell 11
Calculate the classification accuracy on the test set.

```python
answers =  [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
            1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
            1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
            1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 
            0, 0, 0, 0, 0, 0]

correct = sum([round(prob) == answer for prob, answer in zip(probabilities, answers)])
percentage_identified = (correct / len(answers)) * 100
print(f"Model accuracy: {percentage_identified:.2f}%")
```

This is the complete guide to creating and training your CNN model to classify cats and dogs. Let me know if you need further clarification on any steps.
