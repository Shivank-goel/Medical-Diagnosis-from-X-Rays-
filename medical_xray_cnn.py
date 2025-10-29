"""
medical_xray_cnn.py
Single-file pipeline:
 - data pipeline (tf.keras.preprocessing or tf.data)
 - transfer learning (EfficientNetB0)
 - training with EarlyStopping + ModelCheckpoint
 - evaluation & confusion matrix
 - Grad-CAM visualizations for explainability
Requirements:
  tensorflow>=2.10, matplotlib, scikit-learn, opencv-python
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import cv2

# --------------------------
# Config
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")   # Updated to use relative path
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

BATCH_SIZE = 16
IMG_SIZE = (224, 224)
SEED = 42
EPOCHS = 25
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "best_xray_model.h5")
AUTOTUNE = tf.data.AUTOTUNE

# --------------------------
# Check for data directory structure
# --------------------------
def check_data_structure():
    """Check if the required data structure exists"""
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    test_dir = os.path.join(DATA_DIR, "test")
    
    if not all(os.path.exists(d) for d in [train_dir, val_dir, test_dir]):
        print(f"⚠️  Data directories not found!")
        print(f"Please create the following structure in {DATA_DIR}:")
        print("data/")
        print("├── train/")
        print("│   ├── class1/")
        print("│   └── class2/")
        print("├── val/")
        print("│   ├── class1/")
        print("│   └── class2/")
        print("└── test/")
        print("    ├── class1/")
        print("    └── class2/")
        print("\nCreating sample directory structure...")
        
        # Create sample structure
        for split in ['train', 'val', 'test']:
            for class_name in ['normal', 'pneumonia']:  # Common X-ray classes
                class_dir = os.path.join(DATA_DIR, split, class_name)
                os.makedirs(class_dir, exist_ok=True)
        
        print("✅ Sample directory structure created!")
        print("Please add your X-ray images to the appropriate directories.")
        return False
    return True

# --------------------------
# Load datasets (train/val/test)
# --------------------------
def load_datasets():
    """Load datasets if data structure exists"""
    if not check_data_structure():
        return None, None, None, None, 0
    
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    test_dir = os.path.join(DATA_DIR, "test")
    
    # Check if directories have images
    train_images = sum([len(files) for r, d, files in os.walk(train_dir) 
                       if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files)])
    
    if train_images == 0:
        print("⚠️  No images found in data directories!")
        print("Please add X-ray images to the train, val, and test directories.")
        return None, None, None, None, 0
    
    try:
        train_ds = image_dataset_from_directory(
            train_dir,
            labels='inferred',
            label_mode='binary',   # Use binary for 2 classes with sigmoid
            batch_size=BATCH_SIZE,
            image_size=IMG_SIZE,
            shuffle=True,
            seed=SEED
        )
        val_ds = image_dataset_from_directory(
            val_dir,
            labels='inferred',
            label_mode='binary',
            batch_size=BATCH_SIZE,
            image_size=IMG_SIZE,
            shuffle=False,
            seed=SEED
        )
        test_ds = image_dataset_from_directory(
            test_dir,
            labels='inferred',
            label_mode='binary',
            batch_size=BATCH_SIZE,
            image_size=IMG_SIZE,
            shuffle=False,
            seed=SEED
        )
        
        class_names = train_ds.class_names
        num_classes = len(class_names)
        print("Classes:", class_names)
        
        # Prefetch for performance
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        return train_ds, val_ds, test_ds, class_names, num_classes
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None, None, None, 0

# --------------------------
# Data augmentation (on the fly)
# --------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
    # small augmentations — X-rays shouldn't be heavily distorted
], name="data_augmentation")

# --------------------------
# Build model: Transfer learning with EfficientNetB0
# --------------------------
def build_model(num_classes):
    """Build the CNN model with transfer learning"""
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=IMG_SIZE + (3,),
        weights="imagenet"
    )
    base_model.trainable = False  # freeze initially
    
    # Input pipeline
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)  # proper preprocessing
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    
    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        metrics = ["accuracy"]
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        metrics = ["accuracy"]
    
    model = models.Model(inputs, outputs, name="xray_effnet")
    
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    
    return model, base_model, loss_fn, metrics

# --------------------------
# Evaluate model
# --------------------------
def evaluate_model(model, test_ds, class_names, num_classes):
    """Evaluate the model and generate reports"""
    print("\nEvaluating on test set...")
    results = model.evaluate(test_ds)
    print("Test results (loss, acc):", results)
    
    # Build predictions and classification report
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        if num_classes == 2:
            preds_label = (preds.ravel() > 0.5).astype(int)
            true_label = labels.numpy().astype(int)  # For binary labels
        else:
            preds_label = np.argmax(preds, axis=1)
            true_label = np.argmax(labels.numpy(), axis=1)
        y_pred.extend(preds_label.tolist())
        y_true.extend(true_label.tolist())
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    # Save confusion matrix
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to {cm_path}")

# --------------------------
# Grad-CAM implementation
# --------------------------
def get_img_array(img_path, size):
    """Load image and preprocess to model input shape"""
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = tf.keras.applications.efficientnet.preprocess_input(array)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Produces Grad-CAM heatmap for a single preprocessed image array
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients of top predicted class w.r.t. conv layer outputs
    grads = tape.gradient(class_channel, conv_outputs)

    # Compute guided weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Superimpose heatmap on original image and show/save
    """
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * alpha + img
    superimposed_img = np.uint8(superimposed_img / superimposed_img.max() * 255)
    # Display
    plt.figure(figsize=(6, 6))
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title(f"Grad-CAM: {os.path.basename(img_path)}")
    plt.show()
    # Save
    cv2.imwrite(cam_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

def generate_gradcam_visualizations(model, test_ds, class_names, num_classes):
    """Generate Grad-CAM visualizations for sample images"""
    # Find a conv layer name to use for Grad-CAM (last conv in base model)
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            last_conv_layer_name = layer.name
            break
    
    if last_conv_layer_name is None:
        # fallback: look in submodels
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, layers.Conv2D):
                        last_conv_layer_name = sublayer.name
                        break
                if last_conv_layer_name:
                    break
    
    if last_conv_layer_name is None:
        print("⚠️  Could not find a convolutional layer for Grad-CAM")
        return
    
    print("Using last conv layer:", last_conv_layer_name)
    
    # Example usage: pick several test images and show Grad-CAM
    test_dir = os.path.join(DATA_DIR, "test")
    sample_paths = []
    for root, dirs, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                sample_paths.append(os.path.join(root, f))
                if len(sample_paths) >= 6:
                    break
        if len(sample_paths) >= 6:
            break
    
    if not sample_paths:
        print("⚠️  No test images found for Grad-CAM visualization")
        return
    
    print(f"Generating Grad-CAM for {len(sample_paths)} sample images...")
    
    for i, p in enumerate(sample_paths):
        try:
            img_arr = get_img_array(p, IMG_SIZE)
            preds = model.predict(img_arr, verbose=0)
            if num_classes == 2:
                pred_idx = int(preds.ravel() > 0.5)
            else:
                pred_idx = np.argmax(preds[0])
            heatmap = make_gradcam_heatmap(img_arr, model, last_conv_layer_name, pred_idx)
            
            cam_filename = f"gradcam_{i+1}_{os.path.basename(p)}"
            cam_path = os.path.join(OUTPUTS_DIR, cam_filename)
            
            print(f"Image: {os.path.basename(p)}")
            print(f"Prediction: {class_names[pred_idx]}, Confidence: {preds.max():.3f}")
            save_and_display_gradcam(p, heatmap, cam_path=cam_path)
            print(f"Grad-CAM saved to {cam_path}\n")
        except Exception as e:
            print(f"Error processing {p}: {e}")

# --------------------------
# Main training function
# --------------------------
def main():
    """Main function to run the training pipeline"""
    print("🔬 Medical X-Ray CNN Diagnosis System")
    print("=" * 50)
    
    # Load datasets
    train_ds, val_ds, test_ds, class_names, num_classes = load_datasets()
    
    if train_ds is None:
        print("❌ Cannot proceed without data. Please add images and try again.")
        return
    
    print(f"✅ Loaded datasets with {num_classes} classes: {class_names}")
    
    # Build model
    print("\n🏗️  Building model...")
    model, base_model, loss_fn, metrics = build_model(num_classes)
    model.summary()
    
    # Callbacks
    es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    mc = callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_loss", save_best_only=True, save_weights_only=False)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    
    # Train (phase 1: frozen base)
    print("\n🚀 Training phase 1: Frozen base model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[es, mc, reduce_lr],
        verbose=1
    )
    
    # Fine-tune: unfreeze some layers
    print("\n🔥 Training phase 2: Fine-tuning...")
    base_model.trainable = True
    
    # We'll unfreeze from this layer onward (tune as needed)
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=loss_fn,
        metrics=metrics
    )
    
    fine_tune_epochs = 10
    total_epochs = len(history.history['loss']) + fine_tune_epochs
    
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=len(history.history['loss']),
        callbacks=[es, mc, reduce_lr],
        verbose=1
    )
    
    # Evaluate on test set
    evaluate_model(model, test_ds, class_names, num_classes)
    
    # Generate Grad-CAM visualizations
    print("\n🔍 Generating Grad-CAM visualizations...")
    generate_gradcam_visualizations(model, test_ds, class_names, num_classes)
    
    # Save final model
    final_model_path = os.path.join(MODELS_DIR, "final_xray_model.h5")
    model.save(final_model_path)
    print(f"✅ Saved final model to {final_model_path}")
    
    print("\n🎉 Training completed successfully!")
    print(f"📁 Outputs saved in: {OUTPUTS_DIR}")
    print(f"🧠 Models saved in: {MODELS_DIR}")

if __name__ == "__main__":
    main()
