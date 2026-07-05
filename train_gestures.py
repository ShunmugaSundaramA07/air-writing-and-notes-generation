import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse

def build_mlp(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="gestures.csv", help="Input CSV file")
    parser.add_argument("--out", type=str, default="models/gesture_model.h5", help="Output model path")
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(__file__), args.data)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset {data_path} not found. Please run collect_gestures.py first.")

    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1).values
    y = df['label'].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    num_classes = len(le.classes_)
    print(f"[INFO] Found {num_classes} classes: {le.classes_}")

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = build_mlp(X.shape[1], num_classes)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    ]

    print("[INFO] Training model...")
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)

    # Save model and labels
    os.makedirs(os.path.dirname(os.path.join(os.path.dirname(__file__), args.out)), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), args.out)
    model.save(out_path)
    
    labels_path = os.path.join(os.path.dirname(out_path), "gesture_labels.npy")
    np.save(labels_path, le.classes_)
    
    print(f"[DONE] Model saved to {out_path}")
    print(f"[DONE] Labels saved to {labels_path}")

if __name__ == "__main__":
    main()
