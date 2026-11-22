import joblib
import keras
from keras import layers, models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

X_img = np.load("X_dataset_multicanal.npy")
X_nir = np.load("X_nir_24.npy")
X_brix = np.load("X_brix_4.npy")
y = np.load("y_tabulares.npy")

part = np.load("particiones.npz")
idx_train, idx_val, idx_test = (
    part["idx_train"],
    part["idx_val"],
    part["idx_test"],
)

scaler_nir = StandardScaler().fit(X_nir[idx_train])
scaler_brix = StandardScaler().fit(X_brix[idx_train])
joblib.dump(scaler_nir, "scaler_nir_mm.joblib")
joblib.dump(scaler_brix, "scaler_brix_mm.joblib")

nir_tr, nir_va, nir_te = (
    scaler_nir.transform(X_nir[idx_train]),
    scaler_nir.transform(X_nir[idx_val]),
    scaler_nir.transform(X_nir[idx_test]),
)
brix_tr, brix_va, brix_te = (
    scaler_brix.transform(X_brix[idx_train]),
    scaler_brix.transform(X_brix[idx_val]),
    scaler_brix.transform(X_brix[idx_test]),
)
img_tr, img_va, img_te = X_img[idx_train], X_img[idx_val], X_img[idx_test]
y_tr, y_va, y_te = y[idx_train], y[idx_val], y[idx_test]

clases = np.unique(y_tr)
pesos = compute_class_weight("balanced", classes=clases, y=y_tr)
class_weight = {int(c): float(w) for c, w in zip(clases, pesos)}

# Rama 1: Imagen (RGB + UV + NIR)
in_img = layers.Input(shape=img_tr.shape[1:], name="in_imagen")
x_img = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(in_img)
x_img = layers.MaxPooling2D((2, 2))(x_img)
x_img = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x_img)
x_img = layers.MaxPooling2D((2, 2))(x_img)
x_img = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x_img)
feat_img = layers.GlobalAveragePooling2D(name="feat_img")(x_img)

# Rama 2: Espectro AS7263 (24 variables)
in_nir = layers.Input(shape=(nir_tr.shape[1],), name="in_nir")
x_nir = layers.Dense(64, activation="relu")(in_nir)
x_nir = layers.Dropout(0.3)(x_nir)
feat_nir = layers.Dense(32, activation="relu", name="feat_nir")(x_nir)

# Rama 3: °Brix (4 puntos)
in_brix = layers.Input(shape=(brix_tr.shape[1],), name="in_brix")
x_brix = layers.Dense(16, activation="relu")(in_brix)
feat_brix = layers.Dense(8, activation="relu", name="feat_brix")(x_brix)

# Fusión y clasificación
fusion = layers.Concatenate(name="capa_fusion")([feat_img, feat_nir, feat_brix])
f = layers.Dense(64, activation="relu")(fusion)
f = layers.Dropout(0.4)(f)
f = layers.Dense(32, activation="relu")(f)
out = layers.Dense(1, activation="sigmoid", name="salida_multimodal")(f)

modelo_mm = models.Model(
    inputs=[in_img, in_nir, in_brix],
    outputs=out,
    name="Modelo_Multimodal_Valencia",
)
modelo_mm.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")],
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
]

modelo_mm.fit(
    [img_tr, nir_tr, brix_tr],
    y_tr,
    validation_data=([img_va, nir_va, brix_va], y_va),
    epochs=80,
    batch_size=8,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)

probs_te = modelo_mm.predict([img_te, nir_te, brix_te], verbose=0).ravel()
preds_te = (probs_te >= 0.5).astype(int)

print("=== EVALUACIÓN MODELO MULTIMODAL (TEST) ===")
print("Matriz de confusión:\n", confusion_matrix(y_te, preds_te))
print(f"F1-score: {f1_score(y_te, preds_te, zero_division=0):.4f}")
if len(np.unique(y_te)) > 1:
  print(f"ROC-AUC:  {roc_auc_score(y_te, probs_te):.4f}")
print(classification_report(y_te, preds_te, digits=4, zero_division=0))

modelo_mm.save("modelo_multimodal_valencia.keras")