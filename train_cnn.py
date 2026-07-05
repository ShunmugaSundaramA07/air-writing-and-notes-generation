import os
import argparse
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def load_dataset(root):
    labels = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    X, y = [], []
    for li, label in enumerate(labels):
        files = glob(os.path.join(root, label, "*.png"))
        for fp in files:
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None: 
                continue
            # ensure 28x28
            img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)
            img = img.astype("float32") / 255.0
            X.append(img[..., None])  # (28,28,1)
            y.append(li)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, labels

def build_cnn(n_classes):
    inputs = tf.keras.Input(shape=(28,28,1))
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history, out_dir):
    """Save training/validation accuracy and loss plots."""
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history.history['accuracy']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Accuracy Plot ---
    axes[0].plot(epochs, history.history['accuracy'], 'b-o', label='Training Accuracy', markersize=4)
    axes[0].plot(epochs, history.history['val_accuracy'], 'r-o', label='Validation Accuracy', markersize=4)
    axes[0].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # --- Loss Plot ---
    axes[1].plot(epochs, history.history['loss'], 'b-o', label='Training Loss', markersize=4)
    axes[1].plot(epochs, history.history['val_loss'], 'r-o', label='Validation Loss', markersize=4)
    axes[1].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    combined_path = os.path.join(out_dir, 'training_graphs.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] training graphs -> {combined_path}")
    plt.close()

    # --- Separate plots ---
    # Accuracy only
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history['accuracy'], 'b-o', label='Training Accuracy', markersize=4)
    plt.plot(epochs, history.history['val_accuracy'], 'r-o', label='Validation Accuracy', markersize=4)
    plt.title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    acc_path = os.path.join(out_dir, 'accuracy_graph.png')
    plt.savefig(acc_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] accuracy graph -> {acc_path}")
    plt.close()

    # Loss only
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history['loss'], 'b-o', label='Training Loss', markersize=4)
    plt.plot(epochs, history.history['val_loss'], 'r-o', label='Validation Loss', markersize=4)
    plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_path = os.path.join(out_dir, 'loss_graph.png')
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] loss graph -> {loss_path}")
    plt.close()


def plot_confusion_matrix(model, X_val, y_val, labels, out_dir):
    """Save confusion matrix heatmap."""
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    all_labels = list(range(len(labels)))
    cm = confusion_matrix(y_val, y_pred, labels=all_labels)
    report = classification_report(y_val, y_pred, target_names=labels, labels=all_labels, zero_division=0)

    print("\n--- Classification Report ---")
    print(report)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.5), max(8, len(labels) * 0.4)))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] confusion matrix -> {cm_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset folder with A/ B/ ...")
    parser.add_argument("--out", type=str, default="./models/airwrite_model.h5", help="Output model path")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size (default: 64)")
    parser.add_argument("--no_augment", action="store_true", help="Disable data augmentation")
    args = parser.parse_args()

    X, y, labels = load_dataset(args.data)
    if len(X) == 0:
        raise SystemExit("Dataset is empty. Collect samples in dataset/<LETTER>/*.png and retry.")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    model = build_cnn(n_classes=len(labels))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    ]

    if not args.no_augment:
        # Data augmentation to improve robustness
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            fill_mode='constant',
            cval=0.0
        )
        datagen.fit(X_train)
        print("[INFO] Training with data augmentation enabled")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=args.batch_size),
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=2
        )
    else:
        print("[INFO] Training without data augmentation")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=2
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"[SAVED] model -> {args.out}")

    # save labels mapping
    labels_path = os.path.join(os.path.dirname(args.out), "labels.npy")
    np.save(labels_path, np.array(labels))
    print(f"[SAVED] labels -> {labels_path}")

    # Generate training/validation graphs
    graphs_dir = os.path.join(os.path.dirname(args.out), "graphs")
    plot_training_history(history, graphs_dir)
    plot_confusion_matrix(model, X_val, y_val, labels, graphs_dir)

if __name__ == "__main__":
    main()