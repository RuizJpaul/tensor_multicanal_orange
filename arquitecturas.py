"""Definición de las arquitecturas de cada rama (reutilizadas por entrenamiento y CV).

NIR, Brix y CNN conservan exactamente la arquitectura original.
Multimodal: se añadió aumento suave de imagen (igual que la CNN) y Dropout en
las ramas de imagen y °Brix para reducir sobreajuste con ~70 muestras de train.
"""
import keras
from keras import layers, models


def _compilar(modelo):
  modelo.compile(
      optimizer=keras.optimizers.Adam(learning_rate=1e-3),
      loss="binary_crossentropy",
      metrics=["accuracy", keras.metrics.AUC(name="auc")],
  )
  return modelo


def _aumento_suave(nombre: str):
  # Activo solo durante el entrenamiento (fit); en predict no altera la imagen.
  return keras.Sequential(
      [layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.08)],
      name=nombre,
  )


def construir_mlp_nir(n_entradas: int):
  inp = layers.Input(shape=(n_entradas,), name="entrada_nir")
  x = layers.Dense(64, activation="relu", name="nir_dense_1")(inp)
  x = layers.Dropout(0.3, name="nir_dropout_1")(x)
  feat = layers.Dense(32, activation="relu", name="feat_nir")(x)
  x = layers.Dropout(0.2)(feat)
  out = layers.Dense(1, activation="sigmoid", name="salida_nir")(x)
  return _compilar(models.Model(inp, out, name="Modelo_NIR_MLP"))


def construir_mlp_brix(n_entradas: int):
  inp = layers.Input(shape=(n_entradas,), name="entrada_brix")
  x = layers.Dense(16, activation="relu", name="brix_dense_1")(inp)
  feat = layers.Dense(8, activation="relu", name="feat_brix")(x)
  x = layers.Dropout(0.2)(feat)
  out = layers.Dense(1, activation="sigmoid", name="salida_brix")(x)
  return _compilar(models.Model(inp, out, name="Modelo_Brix_MLP"))


def construir_cnn(input_shape=(224, 224, 3)):
  inp = layers.Input(shape=input_shape, name="entrada_imagen")
  x = _aumento_suave("aumento_suave")(inp)
  for filtros in (32, 64, 128):
    x = layers.Conv2D(filtros, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
  feat = layers.GlobalAveragePooling2D(name="feat_cnn")(x)
  x = layers.Dense(64, activation="relu")(feat)
  x = layers.Dropout(0.5)(x)
  out = layers.Dense(1, activation="sigmoid", name="salida_cnn")(x)
  return _compilar(models.Model(inp, out, name="Modelo_CNN_Multicanal"))


def construir_multimodal(input_shape_img, n_nir: int, n_brix: int):
  # Rama 1: Imagen (RGB + UV + NIR)
  in_img = layers.Input(shape=input_shape_img, name="in_imagen")
  x_img = _aumento_suave("aumento_suave_mm")(in_img)
  x_img = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x_img)
  x_img = layers.MaxPooling2D((2, 2))(x_img)
  x_img = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x_img)
  x_img = layers.MaxPooling2D((2, 2))(x_img)
  x_img = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x_img)
  feat_img = layers.GlobalAveragePooling2D(name="feat_img")(x_img)
  feat_img = layers.Dropout(0.3, name="img_dropout")(feat_img)

  # Rama 2: Espectro AS7263
  in_nir = layers.Input(shape=(n_nir,), name="in_nir")
  x_nir = layers.Dense(64, activation="relu")(in_nir)
  x_nir = layers.Dropout(0.3)(x_nir)
  feat_nir = layers.Dense(32, activation="relu", name="feat_nir")(x_nir)

  # Rama 3: °Brix (4 puntos)
  in_brix = layers.Input(shape=(n_brix,), name="in_brix")
  x_brix = layers.Dense(16, activation="relu")(in_brix)
  feat_brix = layers.Dense(8, activation="relu", name="feat_brix")(x_brix)
  feat_brix = layers.Dropout(0.2, name="brix_dropout")(feat_brix)

  # Fusión y clasificación
  fusion = layers.Concatenate(name="capa_fusion")([feat_img, feat_nir, feat_brix])
  f = layers.Dense(64, activation="relu")(fusion)
  f = layers.Dropout(0.4)(f)
  f = layers.Dense(32, activation="relu")(f)
  out = layers.Dense(1, activation="sigmoid", name="salida_multimodal")(f)
  return _compilar(
      models.Model([in_img, in_nir, in_brix], out, name="Modelo_Multimodal_Valencia")
  )
