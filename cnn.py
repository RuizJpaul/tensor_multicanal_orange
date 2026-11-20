import keras
from keras import layers, models
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

print("Cargando los datasets multicanal...")
X_train = np.load("X_dataset_multicanal.npy")
y_train = np.load("y_dataset_etiquetas.npy")

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Distribución de clases (0 vs 1): {np.bincount(y_train)}")

# Calcular pesos para compensar el desbalance (88 vs 12)
clases = np.unique(y_train)
pesos = compute_class_weight(class_weight="balanced", classes=clases, y=y_train)
class_weight_dict = {int(c): float(w) for c, w in zip(clases, pesos)}
print("Pesos de clase:", class_weight_dict)


def construir_cnn(input_shape=(224, 224, 3)):
  model = models.Sequential([
      layers.Input(shape=input_shape),
      # Bloque Convolucional 1
      layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
      layers.MaxPooling2D((2, 2)),
      # Bloque Convolucional 2
      layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
      layers.MaxPooling2D((2, 2)),
      # Bloque Convolucional 3
      layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
      layers.MaxPooling2D((2, 2)),
      # Clasificador
      layers.GlobalAveragePooling2D(),
      layers.Dense(64, activation="relu"),
      layers.Dropout(0.5),
      layers.Dense(1, activation="sigmoid"),
  ])

  model.compile(
      optimizer="adam",
      loss="binary_crossentropy",
      metrics=["accuracy", keras.metrics.AUC(name="auc")],
  )
  return model


modelo = construir_cnn(input_shape=X_train.shape[1:])
modelo.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )
]

print("\nIniciando el entrenamiento de la CNN...")
history = modelo.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=8,
    validation_split=0.2,
    shuffle=True,
    class_weight=class_weight_dict,
    callbacks=callbacks,
)

modelo.save("modelo_granulacion_cnn.keras")
print("\n¡Modelo guardado como 'modelo_granulacion_cnn.keras'!")