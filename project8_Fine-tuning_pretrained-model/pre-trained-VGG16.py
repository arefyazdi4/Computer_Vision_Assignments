
def main():

    # Copying images to training, validation, and test directories
    import os
    import shutil
    import pathlib
    original_dir = pathlib.Path("train")
    base_dir = pathlib.Path("cats_vs_dogs_small")

    # Instantiating a small convnet for dogs vs. cats classification
    from tensorflow import keras
    from tensorflow.keras import layers

    # # Using image_dataset_from_directory to read images
    from tensorflow.keras.utils import image_dataset_from_directory
    train_dataset = image_dataset_from_directory(
        base_dir / "train",
        image_size=(180, 180),
        batch_size=32)
    validation_dataset = image_dataset_from_directory(
        base_dir / "validation",
        image_size=(180, 180),
        batch_size=32)
    test_dataset = image_dataset_from_directory(
        base_dir / "test",
        image_size=(180, 180),
        batch_size=32)

    # FEATURE EXTRACTION TOGETHER WITH DATA AUGMENTATIO
    # Instantiating and freezing the VGG16 convolutional base
    conv_base = keras.applications.vgg16.VGG16(
        weights="imagenet",
        include_top=False)
    conv_base.trainable = False
    print(conv_base.summary())

    #  Adding a data augmentation stage and a classifier to the convolutional base
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
        ]
    )
    inputs = keras.Input(shape=(180, 180, 3))
    x = data_augmentation(inputs)
    x = keras.applications.vgg16.preprocess_input(x)
    x = conv_base(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss="binary_crossentropy",
                  optimizer="rmsprop",
                  metrics=["accuracy"])

    # Evaluating the model on the test set

    # Fine-tuning a pretrained model
    # Freezing all layers until the fourth from the last
    conv_base.trainable = True
    for layer in conv_base.layers[:-4]:
        layer.trainable = False

    # Fine-tuning the model
    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
                  metrics=["accuracy"])
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="fine_tuning.keras",
            save_best_only=True,
            monitor="val_loss")
    ]
    history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=validation_dataset,
        callbacks=callbacks)

    #  evaluate this model on the test data
    model = keras.models.load_model("fine_tuning.keras")
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.3f}")


if __name__ == '__main__':
    main()


