import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

import zipfile

# Replace 'DATASET.zip' with the actual name of your zip file
with zipfile.ZipFile('DATASET.zip', 'r') as zip_ref:     # replace it with ur zipfilename
    zip_ref.extractall('DATASET') # extracts to a directory called 'DATASET'

TRAIN_DIR = '/content/DATASET/train'
TEST_DIR = '/content/DATASET/test'
VALID_DIR = '/content/DATASET/valid'

train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for testing

# Load data from directories (assuming your dataset is organized in subfolders with fruit names)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR ,  # Path to the training data
    target_size=(224, 224),  # Resize images to (224, 224)
    batch_size=32,
    class_mode='categorical'  # Categorical since you have multiple classes (fruits)
)

validation_generator = test_datagen.flow_from_directory(
    VALID_DIR,  # Path to the validation data
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,  # Path to the test data
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Step 3: Build the CNN Model

model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten the 3D output to 1D
    Flatten(),
    
    # Fully connected layers
    Dense(512, activation='relu'),
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(train_generator.num_classes, activation='softmax')  # Output layer (softmax for multi-class)
])

# Step 4: Compile the Model

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',  # Loss function for multi-class classification
              metrics=['accuracy'])

# Display model summary
model.summary()

# Step 5: Train the Model

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=25,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping]
)

# Step 6: Evaluate the Model

# Evaluate on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test accuracy: {test_acc:.4f}")

# Step 7: Plot Training History (Optional)

# Plotting the training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting the training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

