import keras
from keras import models
from keras import layers
from keras.preprocessing import image as k_image
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import cv2

from PIL import UnidentifiedImageError
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Add, Conv2DTranspose, GlobalAveragePooling2D, \
    Dense
from keras.optimizers import Adam
from keras.utils import load_img, img_to_array

latent_dim = 100
img_size = [64, 64]
channels = 3
depth = 64

image_dir = './../../io_project_img/img/'
save_dir = './../../io_project_img/generated'
model_folder = "./../../io_project_img/model"


def remove_transparency(source, background_color):
    source_img = source[:, :, :3]
    source_mask = source[:, :, 3] * (1 / 255.0)
    source_mask = np.repeat(source_mask[:, :, np.newaxis], 3, axis=2)

    background_mask = 1.0 - source_mask

    bg_part = (background_color * (1 / 255.0)) * (background_mask)
    source_part = (source_img * (1 / 255.0)) * (source_mask)

    return np.uint8(cv2.addWeighted(bg_part, 255.0, source_part, 255.0, 0.0))


# img into array (n x width x height x 3) and sets color bg to white
def get_images_old(image_dir):
    images = []
    folder_count = 0

    for subfolder in os.listdir(image_dir):
        subfolder_path = os.path.join(image_dir, subfolder)

        if os.path.isdir(subfolder_path):
            folder_count += 1

            image_list = glob.glob(os.path.join(subfolder_path, '*.png'))

            for image_path in image_list:
                img = cv2.imread(image_path, -1)

                if img.shape[2] == 4:
                    img = remove_transparency(img, 255)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)

                images.append(img)
    images = np.asarray(images) / 255
    return images


def load_images_recursive(directory, img_size=(64, 64), alpha_threshold=255):
    images = []

    def load_image(image_path):
        img = cv2.imread(image_path, -1)

        if img.shape[2] == 4:
            img = remove_transparency(img, alpha_threshold)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)

        images.append(img)

    for root, dirs, files in os.walk(directory):
        print(root)
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                load_image(file_path)

    images = np.asarray(images) / 255.
    return images


def build_discriminator(channels=3, depth=64, learning_rate=0.0001, dropout=0.6,
                        decay=1e-8):
    model = models.Sequential()

    model.add(layers.Conv2D(depth * 1, kernel_size=4, input_shape=(img_size[0], img_size[1], channels), padding="same"))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(depth * 1, kernel_size=4, strides=2, padding="same"))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(depth * 1, kernel_size=4, strides=2, padding="same"))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(depth * 2, kernel_size=4, strides=2, padding="same"))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(depth * 4, kernel_size=4, strides=2, padding="same"))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(depth * 8, kernel_size=4, strides=2, padding="same"))
    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(dropout))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    model_optimizer = keras.optimizers.RMSprop(
        learning_rate=learning_rate,
        decay=decay,
        clipvalue=1.0)

    model.compile(optimizer=model_optimizer, loss='binary_crossentropy')
    return model


def build_generator(channels=3, depth=64, dropout=0.3):
    model = models.Sequential()

    model.add(layers.Dense(4 * 4 * depth * 8, input_shape=(latent_dim,)))
    model.add(layers.Reshape((4, 4, depth * 8)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv2DTranspose(depth * 8, kernel_size=4, strides=2, padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(depth * 4, kernel_size=4, strides=2, padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(depth * 2, kernel_size=4, strides=2, padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(depth, kernel_size=4, strides=2, padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(channels, kernel_size=7, padding="same"))
    model.add(layers.Activation("tanh"))

    return model

def build_gan(latent_dim, generator, discriminator, learning_rate=0.0002, decay=1e-8):
    discriminator.trainable = False
    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    model = keras.models.Model(gan_input, gan_output)

    model_optimizer = keras.optimizers.RMSprop(
        learning_rate=learning_rate,
        decay=decay,
        clipvalue=1.0)

    model.compile(optimizer=model_optimizer, loss='binary_crossentropy')
    return model


def save_data(step, generated_images, save_dir):
    r, c = 6, 6

    imgs = 0.5 * generated_images + 0.5
    imgs = np.clip(imgs, 0, 1)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(imgs[cnt])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(save_dir + "/%d.png" % step)
    plt.close()


def train_gan(discriminator, generator, gan, x_train, iterations, save_dir=None,
              batch_size=64, save_intervals=10, save_weights=True):
    for epoch in range(start_epoch, iterations):
        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
        generated_images = generator.predict(random_latent_vectors)

        real_image_index = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[real_image_index]
        combined_images = np.concatenate([generated_images, real_images])

        labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

        labels += 0.05 * np.random.random(labels.shape)

        d_loss = discriminator.train_on_batch(combined_images, labels)

        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

        misleading_targets = np.zeros((batch_size, 1))
        a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

        if epoch % 1 == 0:
            print(f"Epoka {epoch}, discriminator loss: {d_loss}, generator loss: {a_loss}")
        if epoch % save_intervals == 0:
            if save_dir is not None:
                save_data(epoch, generated_images, save_dir)

            if save_weights:
                filename = "generator_%09d" % epoch
                generator.save(model_folder + f"/{filename}_{epoch}.keras")
                filenamed = "discriminator_%09d" % epoch
                discriminator.save(model_folder + f"/{filenamed}_{epoch}.keras")


iterations = 50000
batch_size = 128
learning_rate_d = 0.00002  # discriminator learning rate
learning_rate_g = 0.0008  # generator learning rate

x_train = load_images_recursive(image_dir, img_size)
print(len(x_train))

discriminator = build_discriminator(learning_rate=learning_rate_d)
generator = build_generator()

# wczytaj najnowszy model
files = glob.glob(os.path.join(model_folder, "generator" + "*" + ".keras"))
if len(files) > 0:
    files.sort()
    file_name = files[-1]
    generator.load_weights(file_name)
    #     get epoch number
    start_epoch = int(file_name.split("_")[-2])

filesD = glob.glob(os.path.join(model_folder, "discriminator" + "*" + ".keras"))
if len(filesD) > 0:
    filesD.sort()
    file_name = filesD[-1]
    discriminator.load_weights(file_name)

gan = build_gan(latent_dim, generator, discriminator, learning_rate=learning_rate_g)

# Train model:
train_gan(discriminator, generator, gan, x_train, iterations, save_dir=save_dir,
          batch_size=batch_size, save_intervals=10, save_weights=True)
