import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths for your organized dataset
train_dir = 'dataset_cnn/train'
val_dir = 'dataset_cnn/validation'

# Update the data generators for RGB images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,  # Typically, vertical flip may not be appropriate
    shear_range=0.2
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    color_mode='rgb',  # Use 'rgb' for MobileNetV2
    batch_size=16,  # Lower batch size for small datasets
    class_mode='binary',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    color_mode='rgb',
    batch_size=16,
    class_mode='binary',
    shuffle=False  # For consistent evaluation ordering
)

# Load the pre-trained MobileNetV2 model (without the top classification layer)
base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False  # Freeze the base model to use as a feature extractor

# Build the transfer learning model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Use a lower learning rate for fine-tuning
initial_lr = 1e-4
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# Define callbacks: EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint('best_model_transfer.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
]

# Train the model
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=callbacks_list
)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_generator)
print("Validation Accuracy (Transfer Learning):", val_acc)
