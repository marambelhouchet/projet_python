import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os
import json

# --- 1. Define Parameters and Paths ---

# These paths must match where you unzipped your dataset
BASE_DIR = 'C:/Users/user/Downloads/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALID_DIR = os.path.join(BASE_DIR, 'valid')

# Model parameters
IMG_SIZE = (224, 224) # MobileNetV2 uses 224x224
BATCH_SIZE = 8   # <-- FURTHER REDUCED BATCH SIZE
EPOCHS = 10 # Start with 10, you can increase this later if needed

print(f"Training directory: {TRAIN_DIR}")
print(f"Validation directory: {VALID_DIR}")

# --- 2. Load and Preprocess Data ---
# Keras will automatically find all the sub-folders
# and label them for you.

try:
    train_dataset = image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=None,
        image_size=IMG_SIZE,
        interpolation="nearest",
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_dataset = image_dataset_from_directory(
        VALID_DIR,
        validation_split=None,
        image_size=IMG_SIZE,
        interpolation="nearest",
        batch_size=BATCH_SIZE,
        shuffle=False
    )
except FileNotFoundError as e:
    print(f"Error: Dataset directory not found. Check the BASE_DIR path.")
    print(f"Details: {e}")
    exit() # Exit if dataset not found

# Get the class names (e.g., "Tomato___Late_blight", "Potato___healthy", etc.)
# This is CRITICAL for your API later.
class_names = train_dataset.class_names
NUM_CLASSES = len(class_names)
print(f"Found {NUM_CLASSES} classes: {class_names}")

# --- 3. Save Class Names ---
# We MUST save this list so your FastAPI service can understand the model's output.
try:
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)
    print("Saved class names to class_names.json")
except Exception as e:
    print(f"Error saving class names: {e}")


# --- 4. Configure Dataset for Performance ---
# These settings help the model run fast by pre-fetching data.
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 5. Create the Model (Transfer Learning) ---

# 1. Load the pre-trained base model (MobileNetV2) without its top classification layer
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)

# 2. Freeze the base model
# We don't want to re-train the parts that already know how to see shapes and edges.
base_model.trainable = False

# 3. Create our new model "head"
# This is the part we will train.
inputs = Input(shape=IMG_SIZE + (3,), name="input_layer")

# Add data augmentation (optional but recommended)
x = tf.keras.layers.RandomFlip("horizontal")(inputs)
x = tf.keras.layers.RandomRotation(0.1)(x)

# Add a preprocessing layer included with MobileNetV2
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

# The (frozen) base model
x = base_model(x, training=False)

# Our custom output layers
x = GlobalAveragePooling2D(name="global_avg_pool")(x)
x = Dropout(0.2, name="dropout_1")(x) # Add dropout for regularization
outputs = Dense(NUM_CLASSES, activation='softmax', name="output_layer")(x)

# 4. Assemble the full model
model = Model(inputs, outputs)

# --- 6. Compile the Model ---
# We use 'SparseCategoricalCrossentropy' because our labels are
# simple integers (0, 1, 2,...) not one-hot vectors.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 7. Train the Model ---

# Save only the best version of the model
checkpoint_callback = ModelCheckpoint(
    filepath="plant_disease_model_best.h5",
    save_best_only=True,
    monitor="val_accuracy",
    verbose=1
)

print("\n--- Starting Model Training ---")

try:
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=validation_dataset,
        callbacks=[checkpoint_callback]
    )
    print("\n--- Training Potentially Complete (check for errors above) ---")

    # --- 8. Save the Final Model ---
    # We save the final model as well (not just the "best" one)
    try:
        model.save("plant_disease_model_final.h5")
        print("Saved final model to plant_disease_model_final.h5")
    except Exception as e:
        print(f"Error saving final model: {e}")


    # --- 9. Plot Training History (Optional) ---
    def plot_history(history):
        # Check if history object and required keys exist
        if history is None or not hasattr(history, 'history') or 'accuracy' not in history.history:
            print("Could not plot history: history object is invalid or missing data.")
            return

        acc = history.history['accuracy']
        val_acc = history.history.get('val_accuracy', None) # Use .get for validation keys
        loss = history.history['loss']
        val_loss = history.history.get('val_loss', None)

        epochs_range = range(len(acc))

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        if val_acc:
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        if val_loss:
            plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        try:
            plt.savefig("training_history.png")
            print("Saved training history plot to training_history.png")
        except Exception as e:
            print(f"Error saving plot: {e}")

    plot_history(history)

except Exception as e:
    print(f"\n--- An error occurred during training: ---")
    print(e)
    # Attempt to save the model even if training failed partway
    try:
        model.save("plant_disease_model_partial.h5")
        print("Saved partial model state to plant_disease_model_partial.h5")
    except Exception as save_e:
        print(f"Could not save partial model: {save_e}")

