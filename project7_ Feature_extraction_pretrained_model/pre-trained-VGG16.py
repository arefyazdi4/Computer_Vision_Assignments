
def main():

    # Copying images to training, validation, and test directories
    import os
    import shutil
    import pathlib
    original_dir = pathlib.Path("train")

    # Instantiating a small convnet for dogs vs. cats classification
    from tensorflow import keras
    from tensorflow.keras import layers

    # # Using image_dataset_from_directory to read images
    from tensorflow.keras.utils import image_dataset_from_directory
    train_dataset = image_dataset_from_directory(
        original_dir / "train",
        image_size=(180, 180),
        batch_size=32)
    validation_dataset = image_dataset_from_directory(
        original_dir / "validation",
        image_size=(180, 180),
        batch_size=32)
    test_dataset = image_dataset_from_directory(
        original_dir / "test",
        image_size=(180, 180),
        batch_size=32)

    # Instantiating the VGG16 convolutional base
    conv_base = keras.applications.vgg16.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(180, 180, 3))

    print(conv_base.summary())

    # FAST FEATURE EXTRACTION WITHOUT DATA AUGMENTATION
    # Extracting the VGG16 features and corresponding labels
    import numpy as np

    def get_features_and_labels(dataset):
        all_features = []
        all_labels = []
        for images, labels in dataset:
            preprocessed_images = keras.applications.vgg16.preprocess_input(
                images)
            features = conv_base.predict(preprocessed_images)
            all_features.append(features)
            all_labels.append(labels)
        return np.concatenate(all_features), np.concatenate(all_labels)

    train_features, train_labels = get_features_and_labels(train_dataset)
    val_features, val_labels = get_features_and_labels(validation_dataset)
    test_features, test_labels = get_features_and_labels(test_dataset)

    print(train_features.shape)  # (2000, 5, 5, 512)

    # Defining and training the densely connected classifier
    inputs = keras.Input(shape=(5, 5, 512))
    x = layers.Flatten()(inputs)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    model.compile(loss="binary_crossentropy",
                  optimizer="rmsprop",
                  metrics=["accuracy"])
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="feature_extraction.keras",
            save_best_only=True,
            monitor="val_loss")
    ]
    history = model.fit(
        train_features, train_labels,
        epochs=20,
        validation_data=(val_features, val_labels),
        callbacks=callbacks)

    # Plotting the results
    import matplotlib.pyplot as plt
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()

    # FEATURE EXTRACTION TOGETHER WITH DATA AUGMENTATIO
    # Instantiating and freezing the VGG16 convolutional base
    conv_base = keras.applications.vgg16.VGG16(
        weights="imagenet",
        include_top=False)
    conv_base.trainable = False

    #  Printing the list of trainable weights before and after freezing
    conv_base.trainable = True
    print("This is the number of trainable weights "
          "before freezing the conv base:", len(conv_base.trainable_weights))
    # 26
    conv_base.trainable = False
    print("This is the number of trainable weights "
          "after freezing the conv base:", len(conv_base.trainable_weights))
    # 0

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

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="feature_extraction_with_data_augmentation.keras",
            save_best_only=True,
            monitor="val_loss")
    ]
    history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=validation_dataset,
        callbacks=callbacks)

    # Evaluating the model on the test set

    test_model = keras.models.load_model(
    "feature_extraction_with_data_augmentation.keras")
    test_loss, test_acc = test_model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.3f}")

if __name__ == '__main__':
    main()
