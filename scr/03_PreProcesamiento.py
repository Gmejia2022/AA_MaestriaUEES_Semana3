"""
03 - Preprocesamiento de Datos
Proyecto: Taller Colaborativo Semana 3, aprendizaje no supervisado (K-means, DBSCAN, PCA y t-SNE)
Maestria en IA - UEES - Semana 3
Grupo #2
Alumnos:
Ingeniero Gonzalo Mejia Alcivar
Ingeniero Jorge Ortiz Merchan
Ingeniero David Perugachi Rojas
"""

# === Importacion de librerias ===

# Manejo de datos
import pandas as pd
import numpy as np

# Preprocesamiento
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Persistencia de modelos
import joblib

# Utilidades
import os
import warnings

warnings.filterwarnings('ignore')

# === Rutas del proyecto ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'Data')
MODELS_DIR = os.path.join(BASE_DIR, 'Models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def cargar_datos_limpios():
    """Carga el dataset limpio generado en la fase anterior."""
    ruta = os.path.join(DATA_DIR, 'E-commerce_limpio.csv')

    if not os.path.exists(ruta):
        print(f"  ERROR: No se encontro {ruta}")
        print("  Ejecute primero: python scr/02_CargaDatos_EDA.py")
        return None, ""

    df = pd.read_csv(ruta)

    lineas = []
    lineas.append("=" * 50)
    lineas.append("  CARGA DEL DATASET LIMPIO")
    lineas.append("=" * 50)
    lineas.append(f"  Filas:    {df.shape[0]}")
    lineas.append(f"  Columnas: {df.shape[1]}")
    lineas.append(f"  Nombres:  {list(df.columns)}")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return df, texto


def codificar_categoricas(df):
    """Codifica variables categoricas seleccionadas usando LabelEncoder."""
    categoricas = {
        'Gender': 'Gender_enc',
        'Customer_Login_type': 'Login_enc',
        'Payment_method': 'Payment_enc',
        'Order_Priority': 'Priority_enc'
    }

    lineas = []
    lineas.append("\n  CODIFICACION DE VARIABLES CATEGORICAS")
    lineas.append("=" * 50)

    encoders = {}
    for col_original, col_nueva in categoricas.items():
        le = LabelEncoder()
        df[col_nueva] = le.fit_transform(df[col_original])
        encoders[col_original] = le

        lineas.append(f"\n  {col_original} -> {col_nueva}:")
        lineas.append("-" * 30)
        mapeo = dict(zip(le.classes_, le.transform(le.classes_)))
        for clase, codigo in mapeo.items():
            lineas.append(f"    {clase:30s} -> {codigo}")

    lineas.append("\n" + "=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return df, encoders, texto


def seleccionar_features(df):
    """Selecciona y separa las features para clustering."""
    features_num = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping_Cost', 'Aging']
    features_enc = ['Gender_enc', 'Login_enc', 'Payment_enc', 'Priority_enc']

    X_num = df[features_num].values
    X_ext = df[features_num + features_enc].values

    lineas = []
    lineas.append("\n  SELECCION DE FEATURES")
    lineas.append("=" * 50)
    lineas.append(f"  Features numericas (primarias): {features_num}")
    lineas.append(f"  Dimensiones: {X_num.shape}")
    lineas.append(f"\n  Features extendidas (num + cat codificadas): {features_num + features_enc}")
    lineas.append(f"  Dimensiones: {X_ext.shape}")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return X_num, X_ext, features_num, features_enc, texto


def escalar_features(X_num, X_ext, features_num, features_enc):
    """Aplica StandardScaler a las features seleccionadas."""
    # Escalar features numericas
    scaler_num = StandardScaler()
    X_num_scaled = scaler_num.fit_transform(X_num)

    # Escalar features extendidas
    scaler_ext = StandardScaler()
    X_ext_scaled = scaler_ext.fit_transform(X_ext)

    # Guardar scalers
    ruta_scaler = os.path.join(MODELS_DIR, 'standard_scaler.pkl')
    joblib.dump(scaler_num, ruta_scaler)

    lineas = []
    lineas.append("\n  ESCALAMIENTO DE FEATURES (StandardScaler)")
    lineas.append("=" * 50)

    lineas.append("\n  Features numericas escaladas:")
    lineas.append("-" * 30)
    df_stats = pd.DataFrame({
        'Feature': features_num,
        'Media_original': X_num.mean(axis=0).round(4),
        'Std_original': X_num.std(axis=0).round(4),
        'Media_escalada': X_num_scaled.mean(axis=0).round(6),
        'Std_escalada': X_num_scaled.std(axis=0).round(6)
    })
    lineas.append(df_stats.to_string(index=False))

    lineas.append(f"\n  Scaler guardado en: {ruta_scaler}")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return X_num_scaled, X_ext_scaled, scaler_num, texto


def guardar_datos_procesados(X_num_scaled, X_ext_scaled, df, features_num, features_enc):
    """Guarda las matrices de features escaladas y datos preprocesados."""
    # Features numericas escaladas
    df_num = pd.DataFrame(X_num_scaled, columns=features_num)
    ruta_num = os.path.join(DATA_DIR, 'features_scaled.csv')
    df_num.to_csv(ruta_num, index=False)
    print(f"  Features escaladas guardadas en: {ruta_num}")

    # Datos preprocesados: eliminar columnas categoricas originales (ya codificadas)
    categoricas_originales = ['Gender', 'Customer_Login_type', 'Payment_method', 'Order_Priority']
    df_pre = df.drop(columns=categoricas_originales)
    ruta_pre = os.path.join(DATA_DIR, 'datos_preprocesados.csv')
    df_pre.to_csv(ruta_pre, index=False)
    print(f"  Datos preprocesados guardados en: {ruta_pre}")
    print(f"  Columnas categoricas originales eliminadas: {categoricas_originales}")
    print(f"  Columnas finales: {list(df_pre.columns)}")


def guardar_reporte(txt_carga, txt_codificacion, txt_features, txt_escalado):
    """Guarda el reporte completo del preprocesamiento."""
    reporte = []
    reporte.append("ETAPA 3: PREPROCESAMIENTO DE DATOS")
    reporte.append("=" * 50)
    reporte.append(txt_carga)
    reporte.append(txt_codificacion)
    reporte.append(txt_features)
    reporte.append(txt_escalado)

    ruta_reporte = os.path.join(RESULTS_DIR, '03_preprocesamiento_reporte.txt')
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write("\n".join(reporte))
    print(f"\n  Reporte guardado en: {ruta_reporte}")


if __name__ == '__main__':
    # 1. Cargar datos limpios
    df, txt_carga = cargar_datos_limpios()
    if df is None:
        exit(1)

    # 2. Codificar variables categoricas
    df, encoders, txt_codificacion = codificar_categoricas(df)

    # 3. Seleccionar features
    X_num, X_ext, features_num, features_enc, txt_features = seleccionar_features(df)

    # 4. Escalar features
    X_num_scaled, X_ext_scaled, scaler, txt_escalado = escalar_features(
        X_num, X_ext, features_num, features_enc
    )

    # 5. Guardar datos procesados
    guardar_datos_procesados(X_num_scaled, X_ext_scaled, df, features_num, features_enc)

    # 6. Guardar reporte
    guardar_reporte(txt_carga, txt_codificacion, txt_features, txt_escalado)
