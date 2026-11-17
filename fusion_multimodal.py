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

def construir_modelo_multimodal_con_mlp(input_shape_img):
    # 1. Rama Espectral (MLP)
    input_espectral = Input(shape=(24,), name="entrada_espectral")
    x1 = Dense(64, activation="relu", name="nir_dense_1")(input_espectral)
    x1 = Dropout(0.3, name="nir_dropout")(x1)
    feat_espectral = Dense(32, activation="relu", name="nir_dense_2")(x1)

    # 2. Rama Físico-Química (Brix)
    input_brix = Input(shape=(1,), name="input_brix")
    x2 = Dense(16, activation="relu", name="brix_dense_1")(input_brix)
    feat_brix = Dense(8, activation="relu", name="brix_dense_2")(x2)

    # 3. Rama Visual (CNN Multicanal optimizada)
    input_imagen = Input(shape=input_shape_img[1:], name="input_imagen")
    x3 = Conv2D(32, (3, 3), activation="relu", padding="same", name="cnn_conv1")(input_imagen)
    x3 = MaxPooling2D((2, 2), name="cnn_pool1")(x3)
    x3 = Conv2D(64, (3, 3), activation="relu", padding="same", name="cnn_conv2")(x3)
    x3 = MaxPooling2D((2, 2), name="cnn_pool2")(x3)
    x3 = Conv2D(128, (3, 3), activation="relu", padding="same", name="cnn_conv3")(x3)
    x3 = GlobalAveragePooling2D(name="cnn_gap")(x3)
    feat_imagen = Dense(32, activation="relu", name="cnn_dense_feat")(x3)

    # 4. Capa de Fusión Multimodal (Concatenación)
    fusion = concatenate([feat_espectral, feat_brix, feat_imagen], name="capa_fusion")
    
    f = Dense(64, activation="relu", name="fusion_dense_1")(fusion)
    f = Dropout(0.4, name="fusion_dropout")(f)
    f = Dense(32, activation="relu", name="fusion_dense_2")(f)
    
    # Salida del Clasificador (Diagnóstico)
    output = Dense(1, activation="sigmoid", name="output_diagnostico")(f)

    modelo = Model(
        inputs=[input_espectral, input_brix, input_imagen],
        outputs=output,
        name="Modelo_Multimodal_Completo"
    )
    
    # Transferencia segura de pesos desde el MLP preentrenado
    try:
        sub_mlp = tf.keras.models.load_model('modelo_espectral_MLP.h5', compile=False)
        # Asignar pesos por correspondencia de capas densas
        modelo.get_layer("nir_dense_1").set_weights(sub_mlp.get_layer("capa_densa_64").get_weights())
        modelo.get_layer("nir_dense_2").set_weights(sub_mlp.get_layer("representacion_espectral").get_weights())
        print("¡Pesos del MLP espectral transferidos exitosamente a la rama NIR!")
    except Exception as e:
        print(f"Aviso: No se pudieron cargar los pesos previos del MLP ({e}). Se inicializarán desde cero.")

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

    print("Construyendo arquitectura multimodal unificada...")
    modelo = construir_modelo_multimodal_con_mlp(X_images.shape)
    
    modelo.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )
    modelo.summary()

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )

    print("Entrenando modelo multimodal con fusión completa...")
    history = modelo.fit(
        [X_spec_train, X_brix_train, X_img_train],
        y_train,
        validation_data=([X_spec_test, X_brix_test, X_img_test], y_test),
        epochs=40,
        batch_size=8,
        callbacks=[early_stopping],
        verbose=1
    )
    
    modelo.save("modelo_multimodal_valencia.h5")
    print("¡Modelo multimodal unificado guardado como 'modelo_multimodal_valencia.h5'!")