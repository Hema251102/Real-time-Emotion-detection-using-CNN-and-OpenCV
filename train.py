import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_emotion_model

print(" Training script started...")

#  Correct dataset paths
train_dir = os.path.join("Dataset", "train")
test_dir = os.path.join("Dataset", "test")

#  Image parameters
img_size = (48, 48)
batch_size = 32

#  Data augmentation for better generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

#  Load datasets
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)

#  Create model
model = create_emotion_model(input_shape=(48, 48, 1), num_classes=len(train_data.class_indices))

#  Compile with optimizer, loss, and metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#  Add callbacks for better training
checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Train model
print(" Starting training...")
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=20,  # Increased for better accuracy
    callbacks=[checkpoint, early_stopping]
)

#  Save final model
model.save('emotion_model.h5')
print(" Model saved as emotion_model.h5")
