#PART A.1 – Load and Explore Image Data
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = 'data/utkface'
IMG_SIZE = 64  # Resize all images to 64x64
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# 1. Parse filenames and extract labels
def load_dataset(data_dir, img_size=64, max_samples=None):
    """
    Load images and extract labels from filenames.
    Filename format: [age]_[gender]_[race]_[date&time].jpg
    Gender: 0=Male, 1=Female
    """
    images = []
    labels = []
    skipped = 0

    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]

    if max_samples:
        image_files = image_files[:max_samples]

    print(f"Total images found: {len(image_files)}")

    for idx, filename in enumerate(image_files):
        try:
            # Parse filename
            parts = filename.split('_')
            if len(parts) < 3:
                skipped += 1
                continue

            age = int(parts[0])
            gender = int(parts[1])

            # Load and preprocess image
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                skipped += 1
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))

            images.append(img)
            labels.append(gender)

            if (idx + 1) % 5000 == 0:
                print(f"Processed {idx + 1} images...")

        except Exception:
            skipped += 1
            continue

    print("\n--- Dataset Loading Complete ---")
    print(f"Successfully loaded: {len(images)} images")
    print(f"Skipped (corrupted/invalid): {skipped} images")

    return np.array(images), np.array(labels)

# 2. Load dataset
print("Loading UTKFace dataset...")
X, y = load_dataset(DATA_DIR, img_size=IMG_SIZE, max_samples=5000)

# 3. Dataset analysis
print(f"\n--- Dataset Statistics ---")
print(f"Images shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Gender distribution: {np.bincount(y)}")
print(f"Male (0): {np.sum(y == 0)}")
print(f"Female (1): {np.sum(y == 1)}")
print(f"Image dtype: {X.dtype}")
print(f"Pixel value range: [{X.min()}, {X.max()}]")

# 4. Visualize sample images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for idx in range(10):
    row = idx // 5
    col = idx % 5
    ax = axes[row, col]
    ax.imshow(X[idx])
    gender_label = "Female" if y[idx] == 1 else "Male"
    ax.set_title(f"Gender: {gender_label}")
    ax.axis('off')

plt.tight_layout()
plt.savefig('sample_images.png', dpi=100, bbox_inches='tight')
plt.show()

# PART A.2 — Normalize and Split Data
from sklearn.model_selection import train_test_split

# 1. Normalize pixel values
print("Normalizing pixel values...")
X_normalized = X.astype('float32') / 255.0
print(f"Normalized pixel range: [{X_normalized.min()}, {X_normalized.max()}]")

# 2. Split into train / val / test
print("\nSplitting dataset...")

# Test set (10%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_normalized, y, test_size=TEST_SPLIT, random_state=42, stratify=y
)

# Validation (20% of remaining)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VALIDATION_SPLIT,
    random_state=42, stratify=y_temp
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# 3. Class balance check
print("\nClass distribution:")
print(f"Training -> Male: {np.sum(y_train == 0)}, Female: {np.sum(y_train == 1)}")
print(f"Validation -> Male: {np.sum(y_val == 0)}, Female: {np.sum(y_val == 1)}")
print(f"Test -> Male: {np.sum(y_test == 0)}, Female: {np.sum(y_test == 1)}")

# PART B.1 — Design & Train the CNN / Fully Connected Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# 1. Flatten the training, validation, and test data
print("Flattening image data for fully connected network...")

# (N, 64, 64, 3) → (N, 12288)
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_val_flattened = X_val.reshape(X_val.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

print(f"Original shape: {X_train.shape}")
print(f"Flattened shape: {X_train_flattened.shape}")
print(f"Total input features: {X_train_flattened.shape[1]}")

# 2. Create a Fully Connected Neural Network
print("\nBuilding Fully Connected Neural Network...")

input_dim = X_train_flattened.shape[1]  # 12,288 features

model = models.Sequential([
    # Hidden Layer 1
    layers.Dense(1024, activation='relu', input_dim=input_dim),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Hidden Layer 2
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Hidden Layer 3
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Hidden Layer 4
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    # Hidden Layer 5
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    # Output layer (binary classification)
    layers.Dense(1, activation='sigmoid')
])

# 3. Compile the model
print("\nCompiling model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
)

# 4. Display model architecture
print("\n--- Model Architecture ---")
model.summary()

# Parameter analysis for first layer
print(f"\n--- Parameter Analysis ---")
print(f"Input features: {input_dim}")
print(f"First hidden layer neurons: 1024")
print(f"Parameters in first layer (weights): {input_dim * 1024} + 1024 (bias) = {input_dim * 1024 + 1024}")

# 5. Define callbacks for better training
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'models/best_gender_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# 6. Train the model
print("\n--- Starting Training ---")
history = model.fit(
    X_train_flattened, y_train,
    batch_size=BATCH_SIZE,
    epochs=2,
    validation_data=(X_val_flattened, y_val),
    callbacks=callbacks,
    verbose=1
)

print("\n--- Training Complete ---")

# PART B.2 — Evaluate Model Performance
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Evaluate on test set
print("Evaluating model on test set...")

test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
    X_test_flattened, y_test, verbose=0
)

print(f"\n--- Test Set Performance ---")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")

# 2. Predictions
print("\nGenerating predictions on test set...")

y_pred_proba = model.predict(X_test_flattened)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# 3. Detailed metrics
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Male', 'Female']))

# 4. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# 5. Plot confusion matrix & ROC curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Male', 'Female'],
            yticklabels=['Male', 'Female'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

axes[1].plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
axes[1].plot([0, 1], [0, 1], linestyle='--')
axes[1].set_title('ROC Curve')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()

plt.tight_layout()
plt.savefig('evaluation_metrics.png', dpi=100, bbox_inches='tight')
plt.show()

# 6. Plot training history (accuracy & loss)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Model Accuracy Over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].set_title('Model Loss Over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
plt.show()


# C.1 Save the trained model
import os

# Create directory if not exists
MODEL_DIR = "saved_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Save model in Keras format (recommended)
model.save(os.path.join(MODEL_DIR, "gender_classifier.keras"))

print("Model saved successfully!")


# C.2 Load the saved model
from keras.models import load_model

MODEL_PATH = "saved_model/gender_classifier.keras"

loaded_model = load_model(MODEL_PATH)

print("Model loaded successfully!")


# C.3 Test loaded model
import numpy as np

# Take one sample from validation/test set
sample_image = X_test[0]
sample_image = np.expand_dims(sample_image, axis=0)

prediction = loaded_model.predict(sample_image)
predicted_gender = "Female" if prediction[0][0] > 0.5 else "Male"

print("Predicted Gender:", predicted_gender)

# C.5 FastAPI App (app.py)
from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Gender Classification API")

# Load model
MODEL_PATH = "saved_model/gender_classifier.keras"
model = load_model(MODEL_PATH)

IMG_SIZE = 64

@app.get("/")
def home():
    return {"message": "Gender Classification API is running"}

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict_gender(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)

    prediction = model.predict(image)
    gender = "Female" if prediction[0][0] > 0.5 else "Male"

    return {
        "prediction": gender,
        "confidence": float(prediction[0][0])
    }


# D.1 Imports
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications import MobileNetV2, VGG16
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# D.2 MobileNetV2 Model (Recommended)
IMG_SIZE = 128

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze base model
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

mobilenet_model = Model(inputs=base_model.input, outputs=output)

mobilenet_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

mobilenet_model.summary()


# D.3 Train MobileNetV2
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.2)
]

history = mobilenet_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=callbacks
)

# D.4 Fine-Tuning (Unfreeze last layers)
# Unfreeze last 30 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

mobilenet_model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_fine = mobilenet_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=callbacks
)

# D.5 Evaluate MobileNetV2
test_loss, test_acc = mobilenet_model.evaluate(X_test, y_test)
print("MobileNetV2 Test Accuracy:", test_acc)


#  D.6 VGG16 Model
base_model = VGG16(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

vgg_model = Model(inputs=base_model.input, outputs=output)

vgg_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

vgg_model.summary()

# D.7 Train VGG16
history_vgg = vgg_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    callbacks=callbacks
)

# D.8 Save Transfer Learning Model
mobilenet_model.save("saved_model/gender_mobilenet.keras")
vgg_model.save("saved_model/gender_vgg16.keras")

print("Transfer learning models saved!")

# D.9 Plot Training Curves
def plot_history(history, title):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

plot_history(history, "MobileNetV2 Training")
plot_history(history_fine, "MobileNetV2 Fine-Tuning")
plot_history(history_vgg, "VGG16 Training")