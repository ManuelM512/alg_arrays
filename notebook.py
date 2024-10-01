# %%
# import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread  # , imshow


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


# %% [markdown]
# 1- Cargar dos imágenes y mostrarlas con `imread`

# %%
# Leer las imágenes
img_mclaren = cv2.imread("mclaren.jpg")
img_ferrari = cv2.imread("ferrari.jpg")


# %%
# Mostrarlas con plt
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_mclaren, cv2.COLOR_BGR2RGB))
plt.title("McLaren")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_ferrari, cv2.COLOR_BGR2RGB))
plt.title("Ferrari")
plt.axis("off")

plt.show()

# %% [markdown]
# 2- Imprimir el tamaño de cada imagen
# %%
print("Tamaño de la imagen McLaren:", img_mclaren.shape)
print("Tamaño de la imagen Ferrari:", img_ferrari.shape)

# %% [markdown]
# 3- Recortar ambas imágenes para que tengan el mismo tamaño, a la vez que sean cuadradas

# %%
side_length = min(img_ferrari.shape[:2] + img_mclaren.shape[:2])
starting_x = 400
starting_y = 500
recortar_imagen_v2(
    "mclaren.jpg",
    "mclaren_rec.jpg",
    starting_x,
    starting_x + side_length - 1,  # Como empieza desde 0
    starting_y,
    starting_y + side_length - 1,
)
recortar_imagen_v2(
    "ferrari.jpg",
    "ferrari_rec.jpg",
    0,
    side_length - 1,
    200,
    200 + side_length - 1,
)
mclaren_rec = imread("mclaren_rec.jpg")
ferrari_rec = imread("ferrari_rec.jpg")

# %%
# Mostrar primera imagen
plt.subplot(1, 2, 1)
plt.imshow(mclaren_rec)
plt.title("McLaren")
plt.axis("off")

# Mostrar segunda imagen
plt.subplot(1, 2, 2)
plt.imshow(ferrari_rec)
plt.title("Ferrari")
plt.axis("off")

plt.show()
# %% [markdown]
# ### 4- Mostrar una de las imágenes como matriz (Mclaren)

# %%
print("Matriz de la imagen McLaren:\n", mclaren_rec)
print("Tamaño de la imagen McLaren:", mclaren_rec.shape)
# %% [markdown]
# ### 5- Calcular la matriz traspuesta de las imagenes
# Solo se intercambian las coordenadas x, y (eje 0 y 1), dejando el eje z igual
# %%
img_mclaren_transpose = np.transpose(mclaren_rec, (1, 0, 2))
img_ferrari_transpose = np.transpose(ferrari_rec, (1, 0, 2))
# Mostrar la matriz traspuesta
print("Matriz traspuesta de la imagen McLaren:\n", img_mclaren_transpose)
print("Tamaño de la imagen traspuesta McLaren:", img_mclaren_transpose.shape)

print("Matriz traspuesta de la imagen Ferrari:\n", img_ferrari_transpose)
print("Tamaño de la imagen traspuesta Ferrari:", img_ferrari_transpose.shape)
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
plt.imshow(img_mclaren_transpose)
plt.title("McLaren Traspuesta")
plt.axis("off")

plt.show()
# %%
plt.figure(figsize=(10, 5))

# Mostrar imagen original
plt.subplot(1, 2, 1)
plt.imshow(ferrari_rec)
plt.title("Ferrari Original")
plt.axis("off")

# Mostrar imagen traspuesta
plt.subplot(1, 2, 2)
plt.imshow(img_ferrari_transpose)
plt.title("Ferrari Traspuesta")
plt.axis("off")

plt.show()
# %% [markdown]
# ### 6- Convertir ambas imágenes a escala de grises y mostrar el recorte


# %%
def convert_to_grayscale(img):
    grayscale = np.mean(img, axis=2)

    # De esta manera nos aseguramos que sigue en el rango 0-255
    grayscale_image = grayscale.astype(np.uint8)
    return grayscale_image


# Convertir las imágenes a escala de grises
gray_mclaren = convert_to_grayscale(mclaren_rec)
gray_ferrari = convert_to_grayscale(ferrari_rec)
# %%
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
cv2.imwrite("ferrary_gray.jpg", gray_ferrari)
cv2.imwrite("mclaren_gray.jpg", gray_mclaren)


# %% [markdown]
# ### 7- Verificar si son invertibles y calcular
# %%
def is_invertible(matrix) -> bool:
    det = np.linalg.det(matrix)

    # Si el determinante es distinto de 0, es invertible
    return det != 0


# %%
if is_invertible(gray_mclaren):
    inv_mclaren = np.linalg.inv(gray_mclaren)
if is_invertible(gray_ferrari):
    inv_ferrari = np.linalg.inv(gray_ferrari)


# %% [markdown]
# ### 8- Producto de una matriz por un escalar
#
# Función para ajustar el contraste de una imagen multiplicando por un escalar
# %%
def ajustar_contraste(img, alpha):
    # Multiplicamos la imagen por el escalar
    img_ajustada = img * alpha

    # Usamos np.clip para restringir los valores entre 0 y 255
    img_ajustada = np.clip(img_ajustada, 0, 255).astype(np.uint8)

    return img_ajustada


# %%
def cambiar_contraste_imagen(img_path, alpha1, alpha2):
    img_rgb = cv2.imread(img_path)

    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # CASO 1: α > 1
    img_caso1 = ajustar_contraste(img_rgb, alpha1)

    # CASO 2: 0 < α < 1
    img_caso2 = ajustar_contraste(img_rgb, alpha2)

    # Mostrar la imagen original y los dos casos
    plt.figure(figsize=(15, 5))

    # Imagen original
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Imagen Original")
    plt.axis("off")

    # Imagen con α > 1
    plt.subplot(1, 3, 2)
    plt.imshow(img_caso1)
    plt.title(f"Contraste aumentado (α={alpha1})")
    plt.axis("off")

    # Imagen con 0 < α < 1
    plt.subplot(1, 3, 3)
    plt.imshow(img_caso2)
    plt.title(f"Contraste reducido (α={alpha2})")
    plt.axis("off")

    plt.show()


# %%
# Parámetros de los casos
alpha1 = 1.5  # CASO 1: α > 1
alpha2 = 0.5  # CASO 2: 0 < α < 1
cambiar_contraste_imagen("mclaren_gray.jpg", alpha1, alpha2)

# %% [markdown]
# ### 9- Multiplicación de matrices y conmutatividad


# %%
def generar_w(img):
    # Obtener las dimensiones de la imagen
    filas = img.shape[0]

    # Generar la matriz identidad del tamaño de la imagen
    # (solo cuenta el tamaño de las filas y columnas)
    identidad = np.eye(filas)

    # Voltear la matriz identidad horizontalmente (para obtener la anti-diagonal)
    W = np.fliplr(identidad)
    return W


# %%
w = generar_w(gray_mclaren)
w_x_pic = w @ gray_mclaren
pic_x_w = gray_mclaren @ w

# %%
# Imagen original
plt.subplot(1, 2, 1)
plt.imshow(w_x_pic, cmap="gray")
plt.title("w * imagen")
plt.axis("off")

# Imagen negativa
plt.subplot(1, 2, 2)
plt.imshow(pic_x_w, cmap="gray")
plt.title("imagen * w")
plt.axis("off")

plt.show()


# %% [markdown]
# ### 10 - Calcular el negativo de las imagenes
# %%
def calcular_negativo(img):
    # Crear una matriz de 255 del mismo tamaño que la imagen
    matriz_255 = np.full(img.shape, 255)

    # Calcular el negativo de la imagen restando img de la matriz 255
    negativo_img = matriz_255 - img

    # Asegurarse de que los valores estén dentro del rango [0, 255]
    negativo_img = np.clip(negativo_img, 0, 255).astype(np.uint8)

    return negativo_img


# %%
mc_neg = calcular_negativo(gray_mclaren)
# %%
# Imagen original
plt.subplot(1, 2, 1)
plt.imshow(gray_mclaren, cmap="gray")
plt.title("Imagen Original")
plt.axis("off")

# Imagen negativa
plt.subplot(1, 2, 2)
plt.imshow(mc_neg, cmap="gray")
plt.title("Imagen Negativa")
plt.axis("off")

plt.show()
