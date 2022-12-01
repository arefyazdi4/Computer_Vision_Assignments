
def main():

    # Copying images to training, validation, and test directories
    import os
    import shutil
    import pathlib
    original_dir = pathlib.Path("train")
    new_base_dir = pathlib.Path("cats_vs_dogs_small")

    def make_subset(subset_name, start_index, end_index):
        for category in ("cat", "dog"):
            dir = new_base_dir / subset_name / category
            os.makedirs(dir)
            fnames = [f"{category}.{i}.jpg" for i in range(
                start_index, end_index)]
            for fname in fnames:
                shutil.copyfile(src=original_dir / fname,
                                dst=dir / fname)
    make_subset("train", start_index=0, end_index=1000)
    make_subset("validation", start_index=1000, end_index=1500)
    make_subset("test", start_index=1500, end_index=2500)

    # Instantiating a small convnet for dogs vs. cats classification
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(180, 180, 3))
    x = layers.Rescaling(1./255)(inputs)
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
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

    # Using image_dataset_from_directory to read images
    from tensorflow.keras.utils import image_dataset_from_directory
    train_dataset = image_dataset_from_directory(
        new_base_dir / "train",
        image_size=(180, 180),
        batch_size=32)
    validation_dataset = image_dataset_from_directory(
        new_base_dir / "validation",
        image_size=(180, 180),
        batch_size=32)
    test_dataset = image_dataset_from_directory(
        new_base_dir / "test",
        image_size=(180, 180),
        batch_size=32)

    #  Fitting the model using a Dataset
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="convnet_from_scratch_val_loss.keras",
            save_best_only=True,
            monitor="val_loss"),
        keras.callbacks.ModelCheckpoint(
            filepath="convnet_from_scratch_val_accuracy.keras",
            save_best_only=True,
            monitor="val_accuracy")
    ]

    history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=validation_dataset,
        callbacks=callbacks)

    # Displaying curves of loss and accuracy during training
    import matplotlib.pyplot as plt
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

    #  Evaluating the model on the test set
    test_model_acc = keras.models.load_model("convnet_from_scratch_val_accuracy.keras")
    test_model_loss = keras.models.load_model("convnet_from_scratch_val_loss.keras")
    test_loss, test_acc = test_model_loss.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.3f}")


if __name__ == '__main__':
    main()
