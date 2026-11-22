import os
import keras
from keras import layers, models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib

X_brix = np.load("X_brix_4.npy")
y = np.load("y_tabulares.npy")
part = np.load("particiones.npz")
idx_train, idx_val, idx_test = (
    part["idx_train"],
    part["idx_val"],
    part["idx_test"],
)

# Escalar SOLO con train
scaler = StandardScaler()
X_train = scaler.fit_transform(X_brix[idx_train])
X_val = scaler.transform(X_brix[idx_val])
X_test = scaler.transform(X_brix[idx_test])

y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

clases = np.unique(y_train)
pesos = compute_class_weight("balanced", classes=clases, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(clases, pesos)}

inp = layers.Input(shape=(X_train.shape[1],), name="entrada_brix")
x = layers.Dense(16, activation="relu", name="brix_dense_1")(inp)
feat = layers.Dense(8, activation="relu", name="feat_brix")(x)
x = layers.Dropout(0.2)(feat)
out = layers.Dense(1, activation="sigmoid", name="salida_brix")(x)

modelo_brix = models.Model(inputs=inp, outputs=out, name="Modelo_Brix_MLP")
modelo_brix.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")],
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
]

modelo_brix.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=8,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=0,
)

probs_test = modelo_brix.predict(X_test, verbose=0).ravel()
preds_test = (probs_test >= 0.5).astype(int)

print("=== EVALUACIÓN RAMA BRIX (TEST) ===")
print("Matriz de confusión:\n", confusion_matrix(y_test, preds_test))
print(f"F1-score: {f1_score(y_test, preds_test, zero_division=0):.4f}")
if len(np.unique(y_test)) > 1:
  print(f"ROC-AUC:  {roc_auc_score(y_test, probs_test):.4f}")
print(
    classification_report(
        y_test, preds_test, digits=4, zero_division=0
    )
)

modelo_brix.save("modelo_brix_mlp.keras")

joblib.dump(scaler, "scaler_brix.joblib")