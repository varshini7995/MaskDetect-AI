import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Define constants
INIT_LR = 1e-4
EPOCHS = 10
BS = 32
DIRECTORY = "dataset"

print("[INFO] preparing data generators...")
# Data augmentation and loading directly from directory in batches
train_aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2,
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_generator = train_aug.flow_from_directory(
    DIRECTORY,
    target_size=(224, 224),
    batch_size=BS,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_aug.flow_from_directory(
    DIRECTORY,
    target_size=(224, 224),
    batch_size=BS,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Load the MobileNetV2 network, ensuring the head FC layer sets are off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

# Construct the head model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Place the head FC model on top of the base model
model = Model(inputs=baseModel.input, outputs=headModel)

# Loop over all layers in the base model and freeze them
for layer in baseModel.layers:
    layer.trainable = False

# Compile the model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the head of the network using the generators
print("[INFO] training head...")
H = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BS,
    epochs=EPOCHS)

# Save the model to disk inside the models folder
os.makedirs("models", exist_ok=True)
print("[INFO] saving mask detector model...")
model.save("models/model.h5")
print("[INFO] Model training complete and saved successfully!")