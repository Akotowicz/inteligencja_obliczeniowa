import numpy as np
from matplotlib import pyplot as plt

# tworzymy tablice o wymiarach 128x128x3 (3 kanaly to RGB)
# uzupelnioną zerami = kolor czarny
data = np.zeros((128, 128, 3), dtype=np.uint8)

def draw(img, x, y, color):
    img[x, y] = [color[0], color[1], color[2]]

# zamalowanie 4 pikseli w lewym górnym rogu
draw(data, 5, 5, [7, 34, 255])
draw(data, 6, 6, [7, 34, 230])
draw(data, 5, 6, [100, 255, 77])
draw(data, 6, 5, [100, 255, 77])

# rysowanie kilku figur na obrazku
for i in range(128):
    for j in range(128):
        if (i - 64) ** 2 + (j - 64) ** 2 < 900:
            draw(data, i, j, [130, 200, 55])
        elif i > 100 and j > 100:
            draw(data, i, j, [230, 120, 30])
        elif (i - 15) ** 2 + (j - 110) ** 2 < 25:
            draw(data, i, j, [255, 4, 55])
        elif (i - 15) ** 2 + (j - 110) ** 2 == 25 or (i - 15) ** 2 + (j - 110) ** 2 == 26:
            draw(data, i, j, [255, 4, 55])

# konwersja macierzy na obrazek i wyświetlenie
plt.imshow(data, interpolation='nearest')
plt.title("Zdjecie przed przetworzeniem")
plt.show()

# ##################### Zadanie 1b ####################################

def apply_convolution(img: np.array, kernel: np.array, stride=1):
    # Get the height, width, and number of channels of the image
    height, width, c = img.shape[0], img.shape[1], img.shape[2]

    # Get the height, width, and number of channels of the kernel
    kernel_height, kernel_width = kernel.shape[0], kernel.shape[1]

    # Create a new image of original img size minus the border
    # where the convolution can't be applied
    new_img = np.zeros(((height - kernel_height + 1) // stride, (width - kernel_width + 1)//stride, 3))

    # Loop through each pixel in the image
    # But skip the outer edges of the image
    for i in range(kernel_height // 2, height - kernel_height // 2 - 1, stride):
        for j in range(kernel_width // 2, width - kernel_width // 2 - 1, stride):
            # Extract a window of pixels around the current pixel
            window = img[i - kernel_height // 2: i + kernel_height // 2 + 1,
                     j - kernel_width // 2: j + kernel_width // 2 + 1]

            # Apply the convolution to the window and set the result as the value of the current pixel in the new image
            new_img[i//stride, j//stride, 0] = int((window[:, :, 0] * kernel).sum())
            new_img[i//stride, j//stride, 1] = int((window[:, :, 1] * kernel).sum())
            new_img[i//stride, j//stride, 2] = int((window[:, :, 2] * kernel).sum())

    # Clip values to the range 0-255
    new_img = np.clip(new_img, 0, 255)
    return new_img.astype(np.uint8)


kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # pionowe krawedzie

new_img = apply_convolution(data, kernel, 1)

plt.imshow(new_img, interpolation='nearest')
plt.title("Pionowe krawedzie, stride = 1")
plt.show()

# ##################### Zadanie 1c ####################################
new_img2 = apply_convolution(data, kernel, 2)

plt.imshow(new_img2, interpolation='nearest')
plt.title("Pionowe krawedzie, stride = 2")
plt.show()

# ##################### Zadanie 1d ####################################
kernel2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])  # poziome krawedzie

new_img3 = apply_convolution(data, kernel2, 1)
plt.imshow(new_img3, interpolation='nearest')
plt.title("Poziome krawedzie, stride = 1")
plt.show()

new_img4 = apply_convolution(data, kernel2, 2)
plt.imshow(new_img4, interpolation='nearest')
plt.title("Poziome krawedzie, stride = 2")
plt.show()

# ##################### Zadanie 1e ####################################
kernel3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # poziome ukosne 0 deg

new_img3 = apply_convolution(data, kernel3, 1)
plt.imshow(new_img3, interpolation='nearest')
plt.title("Ukosne 0 deg, stride = 1")
plt.show()

new_img4 = apply_convolution(data, kernel3, 2)
plt.imshow(new_img4, interpolation='nearest')
plt.title("Ukosne 0 deg, stride = 2")
plt.show()

# ########################################################################
kernel4 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])  # poziome ukosne 45 deg

new_img3 = apply_convolution(data, kernel4, 1)
plt.imshow(new_img3, interpolation='nearest')
plt.title("Ukosne 45 deg, stride = 1")
plt.show()

new_img4 = apply_convolution(data, kernel4, 2)
plt.imshow(new_img4, interpolation='nearest')
plt.title("Ukosne 45 deg, stride = 2")
plt.show()

# ########################################################################
kernel5 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # poziome ukosne 90 deg

new_img3 = apply_convolution(data, kernel5, 1)
plt.imshow(new_img3, interpolation='nearest')
plt.title("Ukosne 90 deg, stride = 1")
plt.show()

new_img4 = apply_convolution(data, kernel5, 2)
plt.imshow(new_img4, interpolation='nearest')
plt.title("Ukosne 90 deg, stride = 2")
plt.show()
# ########################################################################
kernel6 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])  # poziome ukosne 135 deg

new_img3 = apply_convolution(data, kernel6, 1)
plt.imshow(new_img3, interpolation='nearest')
plt.title("Ukosne 135 deg, stride = 1")
plt.show()

new_img4 = apply_convolution(data, kernel6, 2)
plt.imshow(new_img4, interpolation='nearest')
plt.title("Ukosne 135 deg, stride = 2")
plt.show()