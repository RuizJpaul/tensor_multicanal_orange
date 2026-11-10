import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def cargar_datos_multimodales():
    df_nir = pd.read_excel('Dataset.xlsx', sheet_name='Dataset_Nir')
    df_brix = pd.read_excel('Dataset.xlsx', sheet_name='Brix')
    
    cols_nir = [c for c in df_nir.columns if c not in ['ID', 'Etiqueta (granulación)']]
    X_spectral = df_nir[cols_nir].values
    X_brix = df_brix[['Promedio']].values

    X_images = np.load('X_dataset_multicanal.npy')
    y = np.load('y_dataset_etiquetas.npy')

    return X_spectral, X_brix, X_images, y

def construir_modelo_multimodal_optimizado(input_shape_img):
    # Rama 1: Espectral (NIR - MLP)
    input_espectral = Input(shape=(24,), name="input_espectral")
    x1 = Dense(64, activation="relu")(input_espectral)
    x1 = Dropout(0.3)(x1)
    feat_espectral = Dense(32, activation="relu", name="feat_espectral")(x1)

    # Rama 2: Físico-Química (Brix - Dense)
    input_brix = Input(shape=(1,), name="input_brix")
    x2 = Dense(16, activation="relu")(input_brix)
    feat_brix = Dense(8, activation="relu", name="feat_brix")(x2)

    # Rama 3: Visual / Fotográfica (CNN Multicanal optimizada con GlobalAveragePooling)
    input_imagen = Input(shape=input_shape_img[1:], name="input_imagen")
    x3 = Conv2D(32, (3, 3), activation="relu", padding="same")(input_imagen)
    x3 = MaxPooling2D((2, 2))(x3)
    x3 = Conv2D(64, (3, 3), activation="relu", padding="same")(x3)
    x3 = MaxPooling2D((2, 2))(x3)
    x3 = Conv2D(128, (3, 3), activation="relu", padding="same")(x3)
    x3 = GlobalAveragePooling2D()(x3) # Reduce parámetros eficientemente
    feat_imagen = Dense(32, activation="relu", name="feat_imagen")(x3)

    # Capa de Fusión Multimodal (Concatenación)
    fusion = concatenate([feat_espectral, feat_brix, feat_imagen], name="capa_fusion")
    
    f = Dense(64, activation="relu")(fusion)
    f = Dropout(0.4)(f)
    f = Dense(32, activation="relu")(f)
    
    output = Dense(1, activation="sigmoid", name="output_diagnostico")(f)

    modelo = Model(
        inputs=[input_espectral, input_brix, input_imagen],
        outputs=output,
        name="Modelo_Multimodal_Optimizado"
    )
    return modelo

if __name__ == "__main__":
    print("Cargando datos multimodales...")
    X_spec, X_brix, X_images, y = cargar_datos_multimodales()

    scaler_spec = StandardScaler()
    X_spec_scaled = scaler_spec.fit_transform(X_spec)

    scaler_brix = StandardScaler()
    X_brix_scaled = scaler_brix.fit_transform(X_brix)

    if X_images.max() > 1.0:
        X_images = X_images.astype('float32') / 255.0

    indices = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y, test_size=0.2, random_state=42, stratify=y
    )

    X_spec_train, X_spec_test = X_spec_scaled[idx_train], X_spec_scaled[idx_test]
    X_brix_train, X_brix_test = X_brix_scaled[idx_train], X_brix_scaled[idx_test]
    X_img_train, X_img_test = X_images[idx_train], X_images[idx_test]

    print("Construyendo arquitectura optimizada...")
    modelo = construir_modelo_multimodal_optimizado(X_images.shape)
    
    modelo.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )
    modelo.summary()

    # Callback para evitar sobreajuste
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )

    print("Entrenando modelo multimodal optimizado...")
    history = modelo.fit(
        [X_spec_train, X_brix_train, X_img_train],
        y_train,
        validation_data=([X_spec_test, X_brix_test, X_img_test], y_test),
        epochs=40,
        batch_size=8,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Guardar el modelo final entrenado para tu tesis
    modelo.save("modelo_multimodal_valencia.h5")
    print("¡Modelo multimodal guardado exitosamente como 'modelo_multimodal_valencia.h5'!")