import os
import keras
from keras import layers, models
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

# Reproducibilidad base
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def calcular_class_weight(y_sub):
  clases = np.unique(y_sub)
  pesos = compute_class_weight(class_weight="balanced", classes=clases, y=y_sub)
  return {int(c): float(w) for c, w in zip(clases, pesos)}


def crear_modelo_nir(input_dim):
  inp = layers.Input(shape=(input_dim,))
  x = layers.Dense(32, activation="relu")(inp)
  x = layers.Dropout(0.3)(x)
  x = layers.Dense(16, activation="relu")(x)
  out = layers.Dense(1, activation="sigmoid")(x)
  m = models.Model(inp, out)
  m.compile(
      optimizer=keras.optimizers.Adam(learning_rate=1e-3),
      loss="binary_crossentropy",
  )
  return m


def crear_modelo_brix(input_dim):
  inp = layers.Input(shape=(input_dim,))
  x = layers.Dense(16, activation="relu")(inp)
  x = layers.Dropout(0.2)(x)
  x = layers.Dense(8, activation="relu")(x)
  out = layers.Dense(1, activation="sigmoid")(x)
  m = models.Model(inp, out)
  m.compile(
      optimizer=keras.optimizers.Adam(learning_rate=1e-3),
      loss="binary_crossentropy",
  )
  return m


def crear_modelo_cnn(input_shape):
  aug = keras.Sequential([
      layers.RandomFlip("horizontal_and_vertical"),
      layers.RandomRotation(0.08),
  ])
  inp = layers.Input(shape=input_shape)
  x = aug(inp)
  x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
  x = layers.MaxPooling2D((2, 2))(x)
  x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
  x = layers.MaxPooling2D((2, 2))(x)
  x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dense(32, activation="relu")(x)
  x = layers.Dropout(0.5)(x)
  out = layers.Dense(1, activation="sigmoid")(x)
  m = models.Model(inp, out)
  m.compile(
      optimizer=keras.optimizers.Adam(learning_rate=1e-3),
      loss="binary_crossentropy",
  )
  return m


def crear_modelo_multimodal(img_shape, nir_dim, brix_dim):
  # Rama 1: Imagen (compacta para evitar sobreajuste en 100 muestras)
  in_img = layers.Input(shape=img_shape, name="in_img")
  x_img = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(in_img)
  x_img = layers.MaxPooling2D((2, 2))(x_img)
  x_img = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x_img)
  x_img = layers.MaxPooling2D((2, 2))(x_img)
  x_img = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x_img)
  feat_img = layers.GlobalAveragePooling2D()(x_img)
  feat_img = layers.Dense(16, activation="relu")(feat_img)

  # Rama 2: NIR
  in_nir = layers.Input(shape=(nir_dim,), name="in_nir")
  x_nir = layers.Dense(32, activation="relu")(in_nir)
  x_nir = layers.Dropout(0.3)(x_nir)
  feat_nir = layers.Dense(16, activation="relu")(x_nir)

  # Rama 3: Brix
  in_brix = layers.Input(shape=(brix_dim,), name="in_brix")
  x_brix = layers.Dense(16, activation="relu")(in_brix)
  feat_brix = layers.Dense(8, activation="relu")(x_brix)

  # Fusión
  fusion = layers.Concatenate()([feat_img, feat_nir, feat_brix])
  f = layers.Dense(32, activation="relu")(fusion)
  f = layers.Dropout(0.4)(f)
  out = layers.Dense(1, activation="sigmoid")(f)

  m = models.Model(inputs=[in_img, in_nir, in_brix], outputs=out)
  m.compile(
      optimizer=keras.optimizers.Adam(learning_rate=5e-4),
      loss="binary_crossentropy",
  )
  return m


def entrenar_y_evaluar_cv(
    tipo_modelo, X_img, X_nir, X_brix, y, n_splits=5, umbral=0.5
):
  skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
  oof_probs = np.zeros(len(y), dtype=np.float32)

  accs, precs, recs, f1s, aucs = [], [], [], [], []
  cm_total = np.zeros((2, 2), dtype=int)

  for fold, (tr_full_idx, te_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
    keras.backend.clear_session()

    # Separar validación interna solo desde tr_full_idx (sin tocar te_idx)
    tr_idx, va_idx = train_test_split(
        tr_full_idx,
        test_size=0.15,
        stratify=y[tr_full_idx],
        random_state=SEED + fold,
    )

    cw = calcular_class_weight(y[tr_idx])
    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        )
    ]

    if tipo_modelo in ("NIR", "MULTIMODAL"):
      sc_nir = StandardScaler().fit(X_nir[tr_idx])
      nir_tr = sc_nir.transform(X_nir[tr_idx])
      nir_va = sc_nir.transform(X_nir[va_idx])
      nir_te = sc_nir.transform(X_nir[te_idx])

    if tipo_modelo in ("BRIX", "MULTIMODAL"):
      sc_brix = StandardScaler().fit(X_brix[tr_idx])
      brix_tr = sc_brix.transform(X_brix[tr_idx])
      brix_va = sc_brix.transform(X_brix[va_idx])
      brix_te = sc_brix.transform(X_brix[te_idx])

    if tipo_modelo == "NIR":
      modelo = crear_modelo_nir(X_nir.shape[1])
      modelo.fit(
          nir_tr,
          y[tr_idx],
          validation_data=(nir_va, y[va_idx]),
          epochs=80,
          batch_size=8,
          class_weight=cw,
          callbacks=cb,
          verbose=0,
      )
      probs = modelo.predict(nir_te, verbose=0).ravel()

    elif tipo_modelo == "BRIX":
      modelo = crear_modelo_brix(X_brix.shape[1])
      modelo.fit(
          brix_tr,
          y[tr_idx],
          validation_data=(brix_va, y[va_idx]),
          epochs=80,
          batch_size=8,
          class_weight=cw,
          callbacks=cb,
          verbose=0,
      )
      probs = modelo.predict(brix_te, verbose=0).ravel()

    elif tipo_modelo == "CNN":
      modelo = crear_modelo_cnn(X_img.shape[1:])
      modelo.fit(
          X_img[tr_idx],
          y[tr_idx],
          validation_data=(X_img[va_idx], y[va_idx]),
          epochs=50,
          batch_size=8,
          class_weight=cw,
          callbacks=cb,
          verbose=0,
      )
      probs = modelo.predict(X_img[te_idx], verbose=0).ravel()

    elif tipo_modelo == "MULTIMODAL":
      modelo = crear_modelo_multimodal(
          X_img.shape[1:], X_nir.shape[1], X_brix.shape[1]
      )
      modelo.fit(
          [X_img[tr_idx], nir_tr, brix_tr],
          y[tr_idx],
          validation_data=([X_img[va_idx], nir_va, brix_va], y[va_idx]),
          epochs=60,
          batch_size=8,
          class_weight=cw,
          callbacks=cb,
          verbose=0,
      )
      probs = modelo.predict([X_img[te_idx], nir_te, brix_te], verbose=0).ravel()

    oof_probs[te_idx] = probs
    preds = (probs >= umbral).astype(int)
    y_te = y[te_idx]

    accs.append(accuracy_score(y_te, preds))
    precs.append(precision_score(y_te, preds, zero_division=0))
    recs.append(recall_score(y_te, preds, zero_division=0))
    f1s.append(f1_score(y_te, preds, zero_division=0))
    if len(np.unique(y_te)) > 1:
      aucs.append(roc_auc_score(y_te, probs))
    cm_total += confusion_matrix(y_te, preds, labels=[0, 1])

  oof_preds = (oof_probs >= umbral).astype(int)
  oof_auc = roc_auc_score(y, oof_probs) if len(np.unique(y)) > 1 else np.nan
  oof_f1 = f1_score(y, oof_preds, zero_division=0)
  oof_rec = recall_score(y, oof_preds, zero_division=0)
  oof_prec = precision_score(y, oof_preds, zero_division=0)

  print(f"\n================ {tipo_modelo} ({n_splits}-FOLD CV) ================")
  print(f"Acc por fold : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
  print(f"Prec por fold: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
  print(f"Rec por fold : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
  print(f"F1 por fold  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
  if aucs:
    print(f"AUC por fold : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
  print(
      f"OOF Global   -> AUC: {oof_auc:.4f} | F1: {oof_f1:.4f} | "
      f"Recall(1): {oof_rec:.4f} | Prec(1): {oof_prec:.4f}"
  )
  print("Matriz de confusión acumulada (OOF):\n", cm_total)

  return oof_probs


def main():
  X_img = np.load("X_dataset_multicanal.npy")
  X_nir = np.load("X_nir_24.npy")
  X_brix = np.load("X_brix_4.npy")
  y = np.load("y_tabulares.npy")

  assert len(X_img) == len(X_nir) == len(X_brix) == len(y), (
      "Los arreglos no tienen la misma cantidad de muestras."
  )

  os.makedirs("resultados_cv", exist_ok=True)
  oof_dict = {"y": y}

  for variante in ["NIR", "BRIX", "CNN", "MULTIMODAL"]:
    oof_probs = entrenar_y_evaluar_cv(
        variante, X_img, X_nir, X_brix, y, n_splits=5, umbral=0.5
    )
    oof_dict[variante.lower()] = oof_probs

  np.savez("resultados_cv/oof_probabilidades.npz", **oof_dict)
  print("\nProbabilidades OOF guardadas en 'resultados_cv/oof_probabilidades.npz'.")


if __name__ == "__main__":
  main()