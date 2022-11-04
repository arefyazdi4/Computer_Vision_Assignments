from tensorflow import keras
from keras.utils import image_dataset_from_directory
from keras import layers
import numpy as np
import os, shutil, pathlib
import matplotlib.pyplot as plt


def main():
    # dir default
    base_dir = pathlib.Path("dataset")
    image_size = (227, 227)
    train_dataset = image_dataset_from_directory(
        base_dir / "train",
        image_size=image_size,
        batch_size=32)
    validation_dataset = image_dataset_from_directory(
        base_dir / "validation",
        image_size=image_size,
        batch_size=32)
    test_dataset = image_dataset_from_directory(
        base_dir / "test",
        image_size=image_size,
        batch_size=32)

    # augmentation
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
        ]
    )

    # create a model
    inputs = keras.Input(shape=(227, 227, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1. / 255)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="rmsprop",
                  metrics=["accuracy"])

    # call back
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="convnet_from_scratch_with_augmentation.keras",
            save_best_only=True,
            monitor="val_loss")
    ]

    # fit model
    history = model.fit(
        train_dataset,
        epochs=3,
        validation_data=validation_dataset,
        callbacks=callbacks)

    # Evalutaing model
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()

    # test accuracy
    test_model = keras.models.load_model(
        "convnet_from_scratch_with_augmentation.keras")
    test_loss, test_acc = test_model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.3f}")


if __name__ == '__main__':
    main()
