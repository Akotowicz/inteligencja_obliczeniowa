from PIL import UnidentifiedImageError
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Add, Conv2DTranspose, GlobalAveragePooling2D, \
    Dense
from keras.optimizers import Adam
from keras.utils import load_img, img_to_array
import numpy as np
import os

# settings
from matplotlib import pyplot as plt

img_size = [64, 64]  # rozmiar obrazka
img_shape = (64, 64, 3)  # rozmiar obrazka
merged_images_folder = "./../../io_project_img/img/emojikitchen"
base_images_folder = "./../../io_project_img/img/emojikitchen_originals"
output_folder = "./../../io_project_img/generated"

epochs = 100
batch_size = 32

co_ile_epoch_zapisuj_generowane_obrazki = 10


# Przygotowanie danych
# bierzemy wszystkie ikonki które są kombinacją ikonek
# i tworzymy zestawienie
# y_train_img => nasza ikonka "kombinowana" np. +1_100.png
# x_train_img1 => +1.png
# x_train_img2 => 100.png


# folder_path => /img (+1_100.png)
# base_img_path => /base_img (+1.png 100.png)
def load_real_samples(folder_path, base_img_path):
    def inner_load_img(file_path):
        # Load image
        img = load_img(file_path, target_size=img_size)
        # Convert image to array
        img_array = img_to_array(img)
        # Scale pixel values to the range [0, 1]
        img_array = img_array / 255.0
        return img_array

    folders_in_folder_path = [folder for folder in os.listdir(folder_path) if
                              os.path.isdir(os.path.join(folder_path, folder))]

    x_train_img1 = []
    x_train_img2 = []
    y_train_img = []

    for x1_name in folders_in_folder_path:
        #     find all images in folder
        x1_folder_path = os.path.join(folder_path, x1_name)
        x1_image_files = [file for file in os.listdir(x1_folder_path) if
                          file.lower().endswith(('.png'))]
        print(x1_name)

        for x2_name in x1_image_files:
            try:
                x2_img_name_without_png = x2_name[:-4]
                y_img = inner_load_img(os.path.join(x1_folder_path, x2_name))

                x1_img = inner_load_img(os.path.join(base_img_path, x1_name + ".png"))
                x2_img = inner_load_img(os.path.join(base_img_path, x2_img_name_without_png + ".png"))

                x_train_img1.append(x1_img)
                x_train_img2.append(x2_img)
                y_train_img.append(y_img)
            except (UnidentifiedImageError, Exception, OSError) as e:
                # print("error for: " + x1_name + " " + x2_name)
                print(e)
                continue
        break
    #     zamiast break może ładować jakieś partie obrazków

    # Convert the list of images to a NumPy array
    return np.array(x_train_img1), np.array(x_train_img2), np.array(y_train_img)


(x1, x2, y) = load_real_samples(merged_images_folder, base_images_folder)

# print(x1[0])
# print(x2[0])
# print(y[0])

print(x1.shape)
print(x2.shape)
print(y.shape)










def build_generator():
    # Wejściowe warstwy dla dwóch zdjęć 64x64
    input_img1 = Input(shape=img_shape)
    input_img2 = Input(shape=img_shape)

    # Kanały zdjęć są łączone
    combined_input = Add()([input_img1, input_img2])

    # Warstwy konwolucyjne
    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(combined_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Warstwy dekonwolucyjne (transponowane)
    x = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Warstwa wyjściowa generująca nowe zdjęcie 64x64
    output_img = Conv2D(3, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

    # Model Keras z dwoma wejściami i jednym wyjściem
    model = Model(inputs=[input_img1, input_img2], outputs=output_img)

    return model




def build_discriminator(input_shape=(64, 64, 3)):
    # Wejściowa warstwa dla zdjęcia 64x64
    input_img = Input(shape=input_shape)

    # Warstwy konwolucyjne
    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(input_img)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Warstwy globalnego uśredniania (global average pooling)
    x = GlobalAveragePooling2D()(x)

    # Warstwa wyjściowa z jednym neuronem, używając funkcji aktywacji sigmoid
    output = Dense(1, activation='sigmoid')(x)

    # Model Keras z jednym wejściem i jednym wyjściem
    model = Model(inputs=input_img, outputs=output)

    return model

# Przykład użycia
generator = build_generator()
generator.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
generator.summary()

# Przykład użycia
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

discriminator.summary()


# Zamrożenie wag dyskryminatora podczas treningu generatora
discriminator.trainable = False

# Wejścia dla dwóch zdjęć
input_img1 = Input(shape=img_shape)
input_img2 = Input(shape=img_shape)

# Generowanie obrazu przez generator
generated_img = generator([input_img1, input_img2])

# Odpowiedź dyskryminatora na wygenerowany obraz
validity = discriminator(generated_img)

# Model GAN - generator łączony z dyskryminatorem
gan = Model(inputs=[input_img1, input_img2], outputs=validity)

# Kompilacja modelu GAN z funkcją kosztu binary_crossentropy i optymalizatorem Adam
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))



def save_generated_images(imgs, epoch):
    # Zapisz wygenerowane obrazy w folderze
    r, c = 2, 2

    # Zapisz wygenerowane obrazy w pliku png
    imgs = 0.5 * imgs + 0.5
    imgs = np.clip(imgs, 0, 1)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(imgs[cnt])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(output_folder + "/%d.png" % epoch)

    plt.close()


# Pętla treningowa
for epoch in range(epochs):

    # ---------------------
    # Trening dyskryminatora
    # ---------------------

    # Wygeneruj przykładowe indeksy dla batcha
    idx = np.random.randint(0, len(x1), batch_size)

    # Wybierz przykładowe pary zdjęć
    imgs1 = x1[idx]
    imgs2 = x2[idx]
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # Wygeneruj obrazy za pomocą generatora
    gen_imgs = generator.predict([imgs1, imgs2])

    # Trenuj dyskryminator na prawdziwych i fałszywych danych
    d_loss_real = discriminator.train_on_batch(imgs1, valid)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ---------------------
    # Trening generatora
    # ---------------------

    # Wygeneruj nowe przykładowe indeksy dla batcha
    idx = np.random.randint(0, len(x1), batch_size)

    # Wybierz przykładowe pary zdjęć
    imgs1 = x1[idx]
    imgs2 = x2[idx]
    valid = np.ones((batch_size, 1))

    # Trenuj generator w ramach modelu GAN
    g_loss = gan.train_on_batch([imgs1, imgs2], valid)

    # ---------------------
    # Wyświetl postęp
    # ---------------------

    # Co kilka epok wyświetl informacje o postępie
    if epoch % 10 == 0:
        print(f"Epoka {epoch}, Długość straty: {d_loss}, Generator Strata: {g_loss}")

    # ---------------------
    # Zapisz wyniki
    # ---------------------

    # Co kilka epok zapisz wygenerowane obrazy
    if epoch % co_ile_epoch_zapisuj_generowane_obrazki == 0:
        # Wybierz kilka przykładów do wygenerowania
        idx = np.random.randint(0, len(x1), 4)
        imgs1 = x1[idx]
        imgs2 = x2[idx]

        # Wygeneruj obrazy za pomocą generatora
        gen_imgs = generator.predict([imgs1, imgs2])

        # Zapisz wygenerowane obrazy
        save_generated_images(gen_imgs, epoch)
