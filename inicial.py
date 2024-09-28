# %%
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imshow


# %%
def recortar_imagen_v2(
    ruta_img: str,
    ruta_img_crop: str,
    x_inicial: int,
    x_final: int,
    y_inicial: int,
    y_final: int,
) -> None:
    """
    Esta función recibe una imagen y devuelve otra imagen recortada.

    Args:
      ruta_img (str): Ruta de la imagen original que se desea recortar.
      ruta_img_crop (str): Ruta donde se guardará la imagen recortada.
      x_inicial (int): Coordenada x inicial del área de recorte.
      x_final (int): Coordenada x final del área de recorte.
      y_inicial (int): Coordenada y inicial del área de recorte.
      y_final (int): Coordenada y final del área de recorte.

    Return
      None
    """
    try:
        # Abrir la imagen
        image = cv2.imread(ruta_img)

        # Obtener la imagen recortada
        image_crop = image[x_inicial:x_final, y_inicial:y_final]

        # Guardar la imagen recortada en la ruta indicada
        cv2.imwrite(ruta_img_crop, image_crop)

        print(
            "Imagen recortada con éxito. El tamaño de la imagen es de"
            + str(image_crop.shape)
        )
    except Exception as e:
        print("Ha ocurrido un error:", str(e))


# %%

# Cargar las imágenes utilizando imread
img_mclaren = cv2.imread("mclaren.jpg")
img_ferrari = cv2.imread("ferrari.jpg")


# %%
# Mostrar las imágenes
plt.figure(figsize=(10, 5))

# Mostrar primera imagen
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_mclaren, cv2.COLOR_BGR2RGB))
plt.title("McLaren")
plt.axis("off")

# Mostrar segunda imagen
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_ferrari, cv2.COLOR_BGR2RGB))
plt.title("Ferrari")
plt.axis("off")

plt.show()

# %%
# Imprimir el tamaño de cada imagen
print("Tamaño de la imagen McLaren:", img_mclaren.shape)
print("Tamaño de la imagen Ferrari:", img_ferrari.shape)

# %%
lado = min(img_ferrari.shape[:2] + img_mclaren.shape[:2])
# revisar de dsp recortarlo mejor
recortar_imagen_v2(
    "mclaren.jpg",
    "mclaren_rec.jpg",
    400,
    400 + lado - 1,
    500,
    500 + lado - 1,
)
recortar_imagen_v2(
    "ferrari.jpg",
    "ferrari_rec.jpg",
    0,
    lado - 1,
    200,
    200 + lado - 1,
)
mclaren_rec = imread("mclaren_rec.jpg")
ferrari_rec = imread("ferrari_rec.jpg")

# %%
# Mostrar una de las imágenes como matriz (Mclaren)
print("Matriz de la imagen McLaren:\n", mclaren_rec)
print("Tamaño de la imagen McLaren:", mclaren_rec.shape)
# %%
# Calcular la matriz traspuesta de la imagen McLaren
# Solo se intercambian las coordenadas x, y (eje 0 y 1), dejando el eje z igual
img_mclaren_transpose = np.transpose(mclaren_rec, (1, 0, 2))

# Mostrar la matriz traspuesta
print("Matriz traspuesta de la imagen McLaren:\n", img_mclaren_transpose)
print("Tamaño de la imagen traspuesta McLaren:", img_mclaren_transpose.shape)
# %%
# Mostrar la imagen original y su traspuesta
plt.figure(figsize=(10, 5))

# Mostrar imagen original
plt.subplot(1, 2, 1)
plt.imshow(mclaren_rec)
plt.title("McLaren Original")
plt.axis("off")

# Mostrar imagen traspuesta
plt.subplot(1, 2, 2)
# TODO:  esa transposicion ahi quedo rara
plt.imshow(cv2.cvtColor(img_mclaren_transpose, cv2.COLOR_BGR2RGB))
plt.title("McLaren Traspuesta")
plt.axis("off")

plt.show()
# %%
# Comentario sobre los resultados:
# La imagen traspuesta intercambia las coordenadas x e y. Esto da como resultado una imagen rotada 90 grados.


# Convertir ambas imágenes a escala de grises y mostrar el recorte
# Recorte (puede ser un área central de 100x100 píxeles)
def convert_to_grayscale(img):
    return np.mean(img, axis=2)


# Convertir las imágenes a escala de grises
gray_mclaren = convert_to_grayscale(mclaren_rec)
gray_ferrari = convert_to_grayscale(ferrari_rec)

# Mostrar las imágenes recortadas en escala de grises
plt.figure(figsize=(10, 10))

# Mostrar McLaren en escala de grises recortada
plt.subplot(1, 2, 1)
plt.imshow(gray_mclaren, cmap="gray")
plt.title("McLaren Gris Recortada")
plt.axis("off")

# Mostrar Ferrari en escala de grises recortada
plt.subplot(1, 2, 2)
plt.imshow(gray_ferrari, cmap="gray")
plt.title("Ferrari Gris Recortada")
plt.axis("off")

plt.show()


# %%
def tiene_inv(matriz):
    # Calcular el determinante
    determinante = np.linalg.det(matriz)

    # Verificar si la matriz tiene inversa
    return determinante != 0


# %%
# calcular inversa
if tiene_inv(gray_mclaren):
    inv_mclaren = np.linalg.inv(gray_mclaren)
if tiene_inv(gray_ferrari):
    inv_ferrari = np.linalg.inv(gray_ferrari)

# %%
inv_mclaren
# %%
inv_ferrari
# %%
