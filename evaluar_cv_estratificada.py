"""Validación cruzada estratificada repetida (estimación honesta del rendimiento).

Con solo 12 granuladas, un único test de 15 muestras (2 granuladas) es muy
inestable. Aquí cada muestra se evalúa como "test" una vez por repetición.

Por cada fold externo:
  1. El fold de entrenamiento se divide (estratificado) en train interno / validación.
  2. El escalado se ajusta SOLO con el train interno (sin fuga de datos).
  3. La validación se usa para EarlyStopping y para elegir el umbral.
  4. Se evalúa en el fold de test.

Se reportan:
  - Métricas OOF (out-of-fold) por repetición: las 100 predicciones juntas, AUC/F1
    más estables. Se resume media ± desviación entre repeticiones.
  - Métricas por fold (más variables, solo de referencia).

Uso:
  python evaluar_cv_estratificada.py                       # todas las ramas
  python evaluar_cv_estratificada.py --ramas nir brix      # solo algunas
  python evaluar_cv_estratificada.py --folds 5 --repeticiones 3
"""
import utilidades_datos as ud  # importar primero (configura TensorFlow)

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import arquitecturas as arq

RAMAS_DISPONIBLES = ["logreg_nir", "logreg_brix", "nir", "brix", "cnn", "multimodal"]
METRICAS = ["roc_auc", "f1_granulada", "recall_granulada", "especificidad", "balanced_accuracy", "accuracy"]


def _escalar(X, idx_fit, *idx_otros):
  sc = StandardScaler().fit(X[idx_fit])
  return [sc.transform(X[i]) for i in (idx_fit, *idx_otros)]


def predecir_fold(rama, datos, y, i_tr, i_va, i_te):
  """Entrena la rama en (i_tr, i_va) y devuelve probabilidades para i_va e i_te."""
  if rama.startswith("logreg_"):
    X = datos["nir"] if rama == "logreg_nir" else datos["brix"]
    X_tr, X_va, X_te = _escalar(X, i_tr, i_va, i_te)
    clf = LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0)
    clf.fit(X_tr, y[i_tr])
    return clf.predict_proba(X_va)[:, 1], clf.predict_proba(X_te)[:, 1]

  if rama == "nir":
    X_tr, X_va, X_te = _escalar(datos["nir"], i_tr, i_va, i_te)
    modelo, epochs = arq.construir_mlp_nir(X_tr.shape[1]), 80
  elif rama == "brix":
    X_tr, X_va, X_te = _escalar(datos["brix"], i_tr, i_va, i_te)
    modelo, epochs = arq.construir_mlp_brix(X_tr.shape[1]), 80
  elif rama == "cnn":
    X = datos["img"]
    X_tr, X_va, X_te = X[i_tr], X[i_va], X[i_te]
    modelo, epochs = arq.construir_cnn(X.shape[1:]), 50
  elif rama == "multimodal":
    n_tr, n_va, n_te = _escalar(datos["nir"], i_tr, i_va, i_te)
    b_tr, b_va, b_te = _escalar(datos["brix"], i_tr, i_va, i_te)
    img = datos["img"]
    X_tr, X_va, X_te = [img[i_tr], n_tr, b_tr], [img[i_va], n_va, b_va], [img[i_te], n_te, b_te]
    modelo, epochs = arq.construir_multimodal(img.shape[1:], n_tr.shape[1], b_tr.shape[1]), 80
  else:
    raise ValueError(f"Rama desconocida: {rama}")

  ud.entrenar(modelo, X_tr, y[i_tr], X_va, y[i_va], epochs=epochs, verbose=0)
  return modelo.predict(X_va, verbose=0).ravel(), modelo.predict(X_te, verbose=0).ravel()


def evaluar_rama(rama, datos, y, ids, folds, repeticiones):
  import keras

  rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeticiones, random_state=ud.SEMILLA)
  filas_fold, filas_oof, preds = [], [], []
  oof_prob = np.zeros((repeticiones, len(y)))
  oof_umbral = np.zeros((repeticiones, len(y)))
  t0 = time.time()

  for k, (i_ext, i_te) in enumerate(rskf.split(np.zeros(len(y)), y)):
    rep, fold = divmod(k, folds)
    ud.fijar_semilla(ud.SEMILLA + k)
    i_tr, i_va = train_test_split(
        i_ext, test_size=0.2, stratify=y[i_ext], random_state=ud.SEMILLA + k
    )
    p_va, p_te = predecir_fold(rama, datos, y, i_tr, i_va, i_te)
    umbral = ud.elegir_umbral(y[i_va], p_va)
    oof_prob[rep, i_te] = p_te
    oof_umbral[rep, i_te] = umbral

    for nombre_u, u in (("0.50", 0.5), ("validacion", umbral)):
      m = ud.calcular_metricas(y[i_te], p_te, u)
      filas_fold.append({"rama": rama, "repeticion": rep, "fold": fold, "tipo_umbral": nombre_u,
                         **{c: m[c] for c in METRICAS}, "umbral": u})
    preds.extend({"rama": rama, "repeticion": rep, "fold": fold, "ID": ids[j], "y_real": int(y[j]),
                  "prob_granulada": float(p), "umbral_validacion": umbral} for j, p in zip(i_te, p_te))
    keras.backend.clear_session()
    print(f"  [{rama}] rep {rep + 1}/{repeticiones} fold {fold + 1}/{folds} "
          f"(umbral val={umbral:.3f}) - {time.time() - t0:.0f}s", flush=True)

  for rep in range(repeticiones):
    p = oof_prob[rep]
    auc_rep = float(roc_auc_score(y, p))
    for nombre_u, pred in (("0.50", (p >= 0.5).astype(int)), ("validacion", (p >= oof_umbral[rep]).astype(int))):
      m = ud.calcular_metricas(y, pred.astype(float), 0.5)
      m["roc_auc"] = auc_rep
      filas_oof.append({"rama": rama, "repeticion": rep, "tipo_umbral": nombre_u,
                        **{c: m[c] for c in METRICAS}, "matriz_confusion": m["matriz_confusion"]})
  return filas_fold, filas_oof, preds


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--ramas", nargs="+", default=RAMAS_DISPONIBLES, choices=RAMAS_DISPONIBLES)
  parser.add_argument("--folds", type=int, default=5)
  parser.add_argument("--repeticiones", type=int, default=3)
  args = parser.parse_args()

  ids, y = ud.cargar_etiquetas()
  datos = {"nir": ud.cargar_nir(ids)[0], "brix": ud.cargar_brix(ids)[0]}
  if {"cnn", "multimodal"} & set(args.ramas):
    datos["img"] = ud.cargar_imagenes(ids, y_ref=y)
  print(f"Datos: n={len(y)} | granuladas={int(y.sum())} | {args.folds} folds x {args.repeticiones} repeticiones")

  ud.RESULTADOS.mkdir(parents=True, exist_ok=True)
  todas_fold, todas_oof = [], []
  for rama in args.ramas:
    print(f"\n>>> Rama: {rama}")
    f_fold, f_oof, preds = evaluar_rama(rama, datos, y, ids, args.folds, args.repeticiones)
    todas_fold += f_fold
    todas_oof += f_oof
    pd.DataFrame(preds).to_csv(ud.RESULTADOS / f"cv_predicciones_{rama}.csv", index=False)

  df_fold, df_oof = pd.DataFrame(todas_fold), pd.DataFrame(todas_oof)
  # Si se evaluó un subconjunto de ramas, se conservan las demás de ejecuciones previas
  for nombre, df in (("cv_por_fold.csv", df_fold), ("cv_oof_por_repeticion.csv", df_oof)):
    ruta = ud.RESULTADOS / nombre
    if ruta.exists():
      previo = pd.read_csv(ruta)
      df = pd.concat([previo[~previo["rama"].isin(args.ramas)], df], ignore_index=True)
    df.to_csv(ruta, index=False)
    if nombre == "cv_oof_por_repeticion.csv":
      df_oof = df

  resumen = df_oof.groupby(["rama", "tipo_umbral"])[METRICAS].agg(["mean", "std"]).round(4)
  resumen.columns = [f"{m}_{s}" for m, s in resumen.columns]
  resumen = resumen.reset_index()
  orden = {r: i for i, r in enumerate(RAMAS_DISPONIBLES)}
  resumen = resumen.sort_values(["tipo_umbral", "rama"], key=lambda s: s.map(orden) if s.name == "rama" else s)
  resumen.to_csv(ud.RESULTADOS / "cv_resumen.csv", index=False)

  print("\n" + "=" * 98)
  print("RESUMEN CV (predicciones out-of-fold; media ± desviación entre repeticiones)")
  print("=" * 98)
  for tipo in ("0.50", "validacion"):
    print(f"\nUmbral: {'fijo 0.50' if tipo == '0.50' else 'elegido en validación interna'}")
    print(f"{'rama':<12} {'ROC-AUC':>15} {'F1 gran.':>15} {'Recall gran.':>15} {'Especif.':>15} {'Bal. Acc':>15}")
    for _, r in resumen[resumen["tipo_umbral"] == tipo].iterrows():
      celdas = [f"{r[f'{m}_mean']:.3f} ± {r[f'{m}_std']:.3f}" for m in
                ("roc_auc", "f1_granulada", "recall_granulada", "especificidad", "balanced_accuracy")]
      print(f"{r['rama']:<12} " + " ".join(f"{c:>15}" for c in celdas))
  print(f"\nArchivos en {ud.RESULTADOS}: cv_resumen.csv, cv_oof_por_repeticion.csv, cv_por_fold.csv, cv_predicciones_*.csv")


if __name__ == "__main__":
  main()
