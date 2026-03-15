import joblib
import yaml
import tensorflow as tf
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from training_analysis import plot_training_history

MODEL_PATH = Path("model.joblib")
CNN_PATH = Path("best_busi_model.keras") 
RANDOM_STATE = 42
# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- Tabular Logic ---
def load_tabular_data(test_size=0.2):
    data = load_breast_cancer()
    return train_test_split(data.data, data.target, test_size=test_size, random_state=RANDOM_STATE)

def train_tabular(save_path=MODEL_PATH, n_estimators=100):
    X_train, X_test, y_train, y_test = load_tabular_data()
    
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    
    joblib.dump({"model": clf, "features": 30}, save_path)
    
    preds = clf.predict(X_test)
    print(f"Tabular Training Complete. Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))

# --- Image Logic (TensorFlow) ---

def build_cnn(learning_rate=1e-4, num_classes=3):
    # 1. Define Data Augmentation layers
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1),
    ])

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        
        # Apply Augmentation & Rescaling
        data_augmentation,
        tf.keras.layers.Rescaling(1./255),
        
        # Block 1
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(), # Helps stabilize training
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Block 2
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Add this between your second MaxPooling and Flatten
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Classifier Head
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5), # Keep this to fight overfitting
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_cnn(train_dir, val_dir, epochs=5, batch_size=32):
    # 1. Load the raw datasets first
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(224, 224), batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=(224, 224), batch_size=batch_size
    )

    # 2. Extract class names BEFORE prefetching
    num_classes = len(train_ds.class_names) 
    print(f"Training on {num_classes} classes.")

    # 3. Apply performance optimizations
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # 4. Build the model using the variable we saved
    model = build_cnn(num_classes=num_classes)
    

    # Use EarlyStopping to prevent overfitting
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[callback])
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=epochs, 
        callbacks=[callback]
    )
    
    # Generate training analysis plots
    plot_training_history(history.history)


    model.save(CNN_PATH)
    print(f"CNN Model saved to {CNN_PATH}")
