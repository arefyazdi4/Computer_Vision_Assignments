import matplotlib.pyplot as  plt
import numpy as np

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.layers.activation import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam


def build_gen(image_shape, zdim):
    model = Sequential()
    model.add(Dense(128, input_dim=zdim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(28 * 28 * 1, activation='tanh'))
    model.add(Reshape(image_shape))
    return model


def build_dis(image_shape):
    model = Sequential()
    model.add(Flatten(input_shape=image_shape))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    return model


def build_gan(gen, dis):
    model = Sequential()
    model.add(gen)
    model.add(dis)
    return model


def show_images(gen):
    z = np.random.normal(0, 1, (16, 100))  # normal dis (avg, variance, size)
    gen_imgs = gen.predict(z)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplot(4, 4, figsize=(4, 4), sharey=True, sharex=True)
    cnt = 0
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmp='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.show()


def train(iterations, batch_size, interval):
    (Xtrain, _), (_, _) = mnist.load_data()
    Xtrain = Xtrain / 127.5 - 1.0
    Xtrain = np.expand_dims(Xtrain, axis=3)

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        ids = np.random.randint(0, Xtrain.shape[0])  # Xtrain index chose random data
        imgs = Xtrain[ids]

        z = np.random.normal(0, 1, (batch_size, 100))  # normal dis (avg, variance, size)
        gen_imgs = gen_v.predict(z)

        dloss_real = dis_v.train_on_batch(imgs, real)
        dloss_fake = dis_v.train_on_batch(imgs, fake)

        dloss, accuracy = 0.5 * np.add(dloss_real, dloss_fake)

        z = np.random.normal(0, 1, (batch_size, 100))  # normal dis (avg, variance, size)
        gloss = gan_v.train_on_batch(z, real)

        if (iteration + 1) % interval == 0:
            losses.append((dloss, gloss))
            accuracies.append(100.0 * accuracy)
            iteration_checks.append(iteration + 1)
            print(f"{iteration + 1} [D loss: {dloss} , acc: {accuracy * 100.0}] [G loss: {gloss}")
            show_images(gen_v)


if __name__ == '__main__':
    img_shape = (28, 28, 1)
    zdim = 100

    dis_v = build_dis(image_shape=img_shape)
    dis_v.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    dis_v.trainable = False
    gen_v = build_gen(img_shape, zdim)
    gan_v = build_gan(gen_v, dis_v)
    gan_v.compile(loss='binary_crossentropy',
                  optimizer=Adam())

    losses = list()
    accuracies = list()
    iteration_checks = list()

    train(5000, 128, 1000)
