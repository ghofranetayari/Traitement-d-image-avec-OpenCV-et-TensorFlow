import cv2
import numpy as np
import tensorflow as tf

# Chargement de l'image
image_path = './test.jpg'
image = cv2.imread(image_path)

# Vérifiez si l'image a été chargée correctement
if image is None:
    print(f"Erreur lors du chargement de l'image à {image_path}")
    exit()

# Affichage de l'image originale
cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Conversion de l'image en niveaux de gris
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Affichage de l'image en niveaux de gris
cv2.imshow('Gray Image', gray_image)
cv2.waitKey(0)

# Redimensionnement de l'image
resized_image = cv2.resize(gray_image, (28, 28))

# Normalisation de l'image pour le modèle TensorFlow
normalized_image = resized_image / 255.0
normalized_image = normalized_image.reshape(-1, 28, 28, 1)



cv2.destroyAllWindows()
