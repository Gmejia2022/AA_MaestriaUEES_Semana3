"""
07 - Prediccion en Produccion
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
import json

# Persistencia de modelos
import joblib

# Utilidades
import os
import warnings

warnings.filterwarnings('ignore')

# === Rutas del proyecto ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'Models')
DATA_DIR = os.path.join(BASE_DIR, 'Data')
PRODUCCION_DIR = os.path.join(BASE_DIR, 'Produccion')

# === Features esperadas por los modelos ===
FEATURES = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping_Cost', 'Aging']

# === Perfiles de K-means (referencia del entrenamiento) ===
PERFILES_KMEANS = {
    0: "Transacciones de bajo valor (Sales~99, Profit~31, Shipping~3)",
    1: "Transacciones de alto valor (Sales~221, Profit~121, Shipping~12)"
}


def cargar_modelos():
    """Carga los modelos entrenados desde la carpeta Models/."""
    modelos = {}
    archivos = {
        'scaler': 'standard_scaler.pkl',
        'kmeans': 'kmeans_model.pkl',
        'dbscan': 'dbscan_model.pkl',
        'pca': 'pca_model.pkl'
    }

    lineas = []
    lineas.append("=" * 50)
    lineas.append("  CARGA DE MODELOS ENTRENADOS")
    lineas.append("=" * 50)

    for nombre, archivo in archivos.items():
        ruta = os.path.join(MODELS_DIR, archivo)
        if not os.path.exists(ruta):
            print(f"  ERROR: No se encontro {ruta}")
            return None, ""
        modelos[nombre] = joblib.load(ruta)
        lineas.append(f"  {nombre:8s} <- {archivo}")

    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return modelos, texto


def cargar_datos_json(ruta_json):
    """Carga datos de prueba desde un archivo JSON."""
    if not os.path.exists(ruta_json):
        print(f"  ERROR: No se encontro {ruta_json}")
        return None, None, ""

    with open(ruta_json, 'r', encoding='utf-8') as f:
        datos = json.load(f)

    transacciones = datos['transacciones']
    df = pd.DataFrame(transacciones)

    lineas = []
    lineas.append("\n  CARGA DE DATOS DE PRODUCCION")
    lineas.append("=" * 50)
    lineas.append(f"  Archivo: {os.path.basename(ruta_json)}")
    lineas.append(f"  Transacciones: {len(transacciones)}")
    lineas.append(f"  Features esperadas: {FEATURES}")

    # Validar que existan todas las features
    faltantes = [f for f in FEATURES if f not in df.columns]
    if faltantes:
        lineas.append(f"  ERROR: Faltan columnas: {faltantes}")
        print("\n".join(lineas))
        return None, None, ""

    lineas.append("  Validacion de features: OK")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return df, transacciones, texto


def predecir(df, modelos):
    """Aplica los modelos a los datos de produccion."""
    X = df[FEATURES].values

    lineas = []
    lineas.append("\n  PREDICCION CON MODELOS ENTRENADOS")
    lineas.append("=" * 50)

    # 1. Escalar features
    X_scaled = modelos['scaler'].transform(X)
    lineas.append("\n  1. Escalamiento (StandardScaler)")
    lineas.append("-" * 30)
    df_scaled = pd.DataFrame(X_scaled, columns=FEATURES)
    lineas.append(df_scaled.to_string(index=False))

    # 2. Prediccion K-means
    clusters_kmeans = modelos['kmeans'].predict(X_scaled)
    df['Cluster_KMeans'] = clusters_kmeans

    lineas.append("\n\n  2. Prediccion K-means")
    lineas.append("-" * 30)
    for i, row in df.iterrows():
        txn_id = row.get('id', f'TXN-{i}')
        cluster = row['Cluster_KMeans']
        perfil = PERFILES_KMEANS.get(cluster, "Perfil desconocido")
        lineas.append(f"  {txn_id}: Cluster {cluster} -> {perfil}")

    # 3. Prediccion DBSCAN (usando distancia a core samples del entrenamiento)
    dbscan = modelos['dbscan']
    X_train = pd.read_csv(os.path.join(DATA_DIR, 'features_scaled.csv')).values
    core_samples = X_train[dbscan.core_sample_indices_]
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]

    clusters_dbscan = []
    for punto in X_scaled:
        distancias = np.linalg.norm(core_samples - punto, axis=1)
        idx_cercano = np.argmin(distancias)
        dist_min = distancias[idx_cercano]
        if dist_min <= dbscan.eps:
            clusters_dbscan.append(core_labels[idx_cercano])
        else:
            clusters_dbscan.append(-1)

    df['Cluster_DBSCAN'] = clusters_dbscan

    lineas.append("\n\n  3. Prediccion DBSCAN")
    lineas.append("-" * 30)
    for i, row in df.iterrows():
        txn_id = row.get('id', f'TXN-{i}')
        cluster = row['Cluster_DBSCAN']
        etiqueta = f"Cluster {cluster}" if cluster >= 0 else "Ruido (no asignado)"
        lineas.append(f"  {txn_id}: {etiqueta}")

    # 4. Proyeccion PCA
    X_pca = modelos['pca'].transform(X_scaled)
    df['PCA_1'] = X_pca[:, 0]
    df['PCA_2'] = X_pca[:, 1]

    lineas.append("\n\n  4. Proyeccion PCA (2D)")
    lineas.append("-" * 30)
    for i, row in df.iterrows():
        txn_id = row.get('id', f'TXN-{i}')
        lineas.append(f"  {txn_id}: PC1={row['PCA_1']:.4f}, PC2={row['PCA_2']:.4f}")

    lineas.append("\n" + "=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return df, texto


def generar_resumen(df):
    """Genera tabla resumen de resultados."""
    lineas = []
    lineas.append("\n  RESUMEN DE PREDICCIONES")
    lineas.append("=" * 50)

    # Tabla resumen
    cols_resumen = ['id', 'nota', 'Sales', 'Profit', 'Cluster_KMeans', 'Cluster_DBSCAN', 'PCA_1', 'PCA_2']
    cols_disponibles = [c for c in cols_resumen if c in df.columns]
    df_resumen = df[cols_disponibles].copy()
    df_resumen['PCA_1'] = df_resumen['PCA_1'].round(4)
    df_resumen['PCA_2'] = df_resumen['PCA_2'].round(4)

    lineas.append(df_resumen.to_string(index=False))

    # Distribucion K-means
    lineas.append("\n\n  Distribucion K-means:")
    lineas.append("-" * 30)
    for cluster in sorted(df['Cluster_KMeans'].unique()):
        n = (df['Cluster_KMeans'] == cluster).sum()
        perfil = PERFILES_KMEANS.get(cluster, "")
        lineas.append(f"  Cluster {cluster}: {n} transacciones -> {perfil}")

    # Distribucion DBSCAN
    lineas.append("\n  Distribucion DBSCAN:")
    lineas.append("-" * 30)
    for cluster in sorted(df['Cluster_DBSCAN'].unique()):
        n = (df['Cluster_DBSCAN'] == cluster).sum()
        etiqueta = f"Cluster {cluster}" if cluster >= 0 else "Ruido"
        lineas.append(f"  {etiqueta}: {n} transacciones")

    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return texto


def guardar_resultados(df, txt_modelos, txt_carga, txt_prediccion, txt_resumen):
    """Guarda resultados de produccion."""
    # CSV con resultados
    ruta_csv = os.path.join(PRODUCCION_DIR, 'resultados_prediccion.csv')
    cols_guardar = ['id', 'nota'] + FEATURES + ['Cluster_KMeans', 'Cluster_DBSCAN', 'PCA_1', 'PCA_2']
    cols_disponibles = [c for c in cols_guardar if c in df.columns]
    df[cols_disponibles].to_csv(ruta_csv, index=False)
    print(f"\n  Resultados guardados en: {ruta_csv}")

    # Reporte de texto
    reporte = []
    reporte.append("ETAPA 7: PREDICCION EN PRODUCCION")
    reporte.append("=" * 50)
    reporte.append(txt_modelos)
    reporte.append(txt_carga)
    reporte.append(txt_prediccion)
    reporte.append(txt_resumen)

    ruta_reporte = os.path.join(PRODUCCION_DIR, 'reporte_produccion.txt')
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write("\n".join(reporte))
    print(f"  Reporte guardado en: {ruta_reporte}")


if __name__ == '__main__':
    # 1. Cargar modelos entrenados
    modelos, txt_modelos = cargar_modelos()
    if modelos is None:
        exit(1)

    # 2. Cargar datos de prueba JSON
    ruta_json = os.path.join(PRODUCCION_DIR, 'datos_prueba.json')
    df, transacciones, txt_carga = cargar_datos_json(ruta_json)
    if df is None:
        exit(1)

    # 3. Predecir con los modelos
    df, txt_prediccion = predecir(df, modelos)

    # 4. Generar resumen
    txt_resumen = generar_resumen(df)

    # 5. Guardar resultados
    guardar_resultados(df, txt_modelos, txt_carga, txt_prediccion, txt_resumen)
