import os
import keras
from keras import layers, models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib

X_nir = np.load("X_nir_24.npy")
y = np.load("y_tabulares.npy")

# Crear o reutilizar partición estratificada fija (70% train / 15% val / 15% test)
if os.path.exists("particiones.npz"):
  part = np.load("particiones.npz")
  idx_train, idx_val, idx_test = (
      part["idx_train"],
      part["idx_val"],
      part["idx_test"],
  )
else:
  idx_all = np.arange(len(y))
  idx_temp, idx_test, _, y_temp = train_test_split(
      idx_all, y, test_size=0.15, stratify=y, random_state=42
  )
  idx_train, idx_val = train_test_split(
      idx_temp, test_size=0.15 / 0.85, stratify=y_temp, random_state=42
  )
  np.savez(
      "particiones.npz", idx_train=idx_train, idx_val=idx_val, idx_test=idx_test
  )

# Escalar SOLO con train
scaler = StandardScaler()
X_train = scaler.fit_transform(X_nir[idx_train])
X_val = scaler.transform(X_nir[idx_val])
X_test = scaler.transform(X_nir[idx_test])

y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

clases = np.unique(y_train)
pesos = compute_class_weight("balanced", classes=clases, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(clases, pesos)}

# MLP funcional (permite extraer 'feat_nir' fácilmente para fusión)
inp = layers.Input(shape=(X_train.shape[1],), name="entrada_nir")
x = layers.Dense(64, activation="relu", name="nir_dense_1")(inp)
x = layers.Dropout(0.3, name="nir_dropout_1")(x)
feat = layers.Dense(32, activation="relu", name="feat_nir")(x)
x = layers.Dropout(0.2)(feat)
out = layers.Dense(1, activation="sigmoid", name="salida_nir")(x)

modelo_nir = models.Model(inputs=inp, outputs=out, name="Modelo_NIR_MLP")
modelo_nir.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")],
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
]

modelo_nir.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=8,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=0,
)

# Evaluación final en TEST (nunca tocado en entrenamiento ni early stopping)
probs_test = modelo_nir.predict(X_test, verbose=0).ravel()
preds_test = (probs_test >= 0.5).astype(int)

print("=== EVALUACIÓN RAMA NIR (TEST) ===")
print("Matriz de confusión:\n", confusion_matrix(y_test, preds_test))
print(f"F1-score: {f1_score(y_test, preds_test, zero_division=0):.4f}")
if len(np.unique(y_test)) > 1:
  print(f"ROC-AUC:  {roc_auc_score(y_test, probs_test):.4f}")
print(
    classification_report(
        y_test, preds_test, digits=4, zero_division=0
    )
)

modelo_nir.save("modelo_nir_mlp.keras")

joblib.dump(scaler, "scaler_nir.joblib")