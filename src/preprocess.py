"""
Preprocessing pipeline: load dataset jadi tf.data.Dataset siap training.

- Training/ di-split jadi train (85%) & val (15%)
- Testing/ dipakai sebagai test set terpisah (tidak disentuh sampai evaluasi akhir)
- Resize ke IMG_SIZE, label di-encode otomatis dari nama folder (alphabetical:
  glioma=0, meningioma=1, notumor=2, pituitary=3)

Import fungsi load_datasets() dari script training nanti, jangan run manual
kecuali mau cek pipeline-nya jalan.
"""

import os

import tensorflow as tf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMG_SIZE = (224, 224)
# Lower batch size for CPU-only Windows runs; the transfer model fine-tuning
# was hitting oneDNN memory-allocation errors with the default 32-image batches.
BATCH_SIZE = 16
SEED = 42


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "Training"),
        validation_split=0.15,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "Training"),
        validation_split=0.15,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "Testing"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,  # penting: urutan tetap, dipakai nanti buat confusion matrix
    )

    class_names = train_ds.class_names
    print(f"Kelas (urutan label): {class_names}")

    # Normalisasi pixel 0-255 -> 0-1
    normalize = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalize(x), y))
    val_ds = val_ds.map(lambda x, y: (normalize(x), y))
    test_ds = test_ds.map(lambda x, y: (normalize(x), y))

    # Prefetch biar training nggak bottleneck di I/O
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def get_augmentation_layer():
    """Dipakai untuk eksperimen 'dengan augmentation' nanti (bukan default)."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])


if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names = load_datasets()
    print("\nCek shape 1 batch:")
    for images, labels in train_ds.take(1):
        print("Images:", images.shape, "Labels:", labels.shape)
    print("\nJumlah batch -> train:", tf.data.experimental.cardinality(train_ds).numpy(),
          "val:", tf.data.experimental.cardinality(val_ds).numpy(),
          "test:", tf.data.experimental.cardinality(test_ds).numpy())
