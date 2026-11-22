import keras
from keras import layers, models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

X = np.load("X_dataset_multicanal.npy")
y = np.load("y_dataset_etiquetas.npy")
part = np.load("particiones.npz")
idx_train, idx_val, idx_test = (
    part["idx_train"],
    part["idx_val"],
    part["idx_test"],
)

X_train, y_train = X[idx_train], y[idx_train]
X_val, y_val = X[idx_val], y[idx_val]
X_test, y_test = X[idx_test], y[idx_test]

clases = np.unique(y_train)
pesos = compute_class_weight("balanced", classes=clases, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(clases, pesos)}

# Aumento suave solo activo en entrenamiento (training=True por defecto en fit)
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.08),
    ],
    name="aumento_suave",
)


def construir_cnn(input_shape=(224, 224, 3)):
  inp = layers.Input(shape=input_shape, name="entrada_imagen")
  x = data_augmentation(inp)

  x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
  x = layers.MaxPooling2D((2, 2))(x)

  x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
  x = layers.MaxPooling2D((2, 2))(x)

  x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
  x = layers.MaxPooling2D((2, 2))(x)

  feat = layers.GlobalAveragePooling2D(name="feat_cnn")(x)
  x = layers.Dense(64, activation="relu")(feat)
  x = layers.Dropout(0.5)(x)
  out = layers.Dense(1, activation="sigmoid", name="salida_cnn")(x)

  model = models.Model(inputs=inp, outputs=out, name="Modelo_CNN_Multicanal")
  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=1e-3),
      loss="binary_crossentropy",
      metrics=["accuracy", keras.metrics.AUC(name="auc")],
  )
  return model


modelo = construir_cnn(input_shape=X_train.shape[1:])
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
]

modelo.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=8,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)

probs_test = modelo.predict(X_test, verbose=0).ravel()
preds_test = (probs_test >= 0.5).astype(int)

print("=== EVALUACIÓN RAMA CNN (TEST) ===")
print("Matriz de confusión:\n", confusion_matrix(y_test, preds_test))
print(f"F1-score: {f1_score(y_test, preds_test, zero_division=0):.4f}")
if len(np.unique(y_test)) > 1:
  print(f"ROC-AUC:  {roc_auc_score(y_test, probs_test):.4f}")
print(classification_report(y_test, preds_test, digits=4, zero_division=0))

modelo.save("modelo_granulacion_cnn.keras")