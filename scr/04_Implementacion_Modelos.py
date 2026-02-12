"""
04 - Implementacion de Modelos de Clustering
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

# Visualizacion
import matplotlib.pyplot as plt
import seaborn as sns

# Clustering
from sklearn.cluster import KMeans, DBSCAN

# Metricas
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Vecinos mas cercanos (para k-distance de DBSCAN)
from sklearn.neighbors import NearestNeighbors

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


def cargar_features():
    """Carga las features escaladas para clustering."""
    ruta = os.path.join(DATA_DIR, 'features_scaled.csv')

    if not os.path.exists(ruta):
        print(f"  ERROR: No se encontro {ruta}")
        print("  Ejecute primero: python scr/03_PreProcesamiento.py")
        return None, ""

    df = pd.read_csv(ruta)
    X = df.values

    lineas = []
    lineas.append("=" * 50)
    lineas.append("  CARGA DE FEATURES ESCALADAS")
    lineas.append("=" * 50)
    lineas.append(f"  Dimensiones: {X.shape}")
    lineas.append(f"  Features: {list(df.columns)}")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return X, texto


def analisis_elbow(X):
    """Determina el numero optimo de clusters usando el metodo del codo."""
    K_range = range(2, 11)
    inertias = []

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        km.fit(X)
        inertias.append(km.inertia_)

    # Grafico del codo
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Numero de Clusters (K)', fontsize=12)
    ax.set_ylabel('Inercia (WCSS)', fontsize=12)
    ax.set_title('Metodo del Codo - K-means', fontsize=14, fontweight='bold')
    ax.set_xticks(list(K_range))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ruta = os.path.join(RESULTS_DIR, '04_elbow_method.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    lineas = []
    lineas.append("\n  METODO DEL CODO (ELBOW METHOD)")
    lineas.append("=" * 50)
    for k, inertia in zip(K_range, inertias):
        lineas.append(f"  K={k:2d}  ->  Inercia = {inertia:.2f}")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return inertias, texto


def analisis_silhouette(X):
    """Evalua el numero optimo de clusters usando el coeficiente de silueta."""
    K_range = range(2, 11)
    scores = []

    # Usar muestra para silhouette (costoso en datasets grandes)
    np.random.seed(42)
    n_muestra = min(10000, len(X))
    idx_muestra = np.random.choice(len(X), n_muestra, replace=False)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        score = silhouette_score(X[idx_muestra], labels[idx_muestra])
        scores.append(score)

    mejor_k = list(K_range)[np.argmax(scores)]
    mejor_score = max(scores)

    # Grafico de silueta
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(K_range), scores, 'rs-', linewidth=2, markersize=8)
    ax.axvline(x=mejor_k, color='green', linestyle='--', linewidth=1.5,
               label=f'Mejor K={mejor_k} (score={mejor_score:.4f})')
    ax.set_xlabel('Numero de Clusters (K)', fontsize=12)
    ax.set_ylabel('Coeficiente de Silueta', fontsize=12)
    ax.set_title('Analisis de Silueta - K-means', fontsize=14, fontweight='bold')
    ax.set_xticks(list(K_range))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ruta = os.path.join(RESULTS_DIR, '04_silhouette_scores.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    lineas = []
    lineas.append("\n  ANALISIS DE SILUETA (SILHOUETTE)")
    lineas.append("=" * 50)
    for k, score in zip(K_range, scores):
        marca = " <-- MEJOR" if k == mejor_k else ""
        lineas.append(f"  K={k:2d}  ->  Silhouette = {score:.4f}{marca}")
    lineas.append(f"\n  K optimo seleccionado: {mejor_k}")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return mejor_k, scores, texto


def entrenar_kmeans(X, n_clusters):
    """Entrena el modelo K-means con el numero optimo de clusters."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X)

    # Guardar modelo
    ruta_modelo = os.path.join(MODELS_DIR, 'kmeans_model.pkl')
    joblib.dump(km, ruta_modelo)

    # Metricas
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)

    lineas = []
    lineas.append("\n  MODELO K-MEANS ENTRENADO")
    lineas.append("=" * 50)
    lineas.append(f"  Numero de clusters: {n_clusters}")
    lineas.append(f"  Inercia: {km.inertia_:.2f}")
    lineas.append(f"  Silhouette Score: {sil:.4f}")
    lineas.append(f"  Calinski-Harabasz Index: {ch:.2f}")
    lineas.append(f"  Davies-Bouldin Index: {db:.4f}")

    lineas.append("\n  Tamanio de clusters:")
    lineas.append("-" * 30)
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        pct = count / len(labels) * 100
        lineas.append(f"    Cluster {cluster}: {count:6d} ({pct:.1f}%)")

    lineas.append(f"\n  Modelo guardado en: {ruta_modelo}")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return labels, km, texto


def optimizar_dbscan(X):
    """Determina los parametros optimos de DBSCAN usando k-distance."""
    k = min(2 * X.shape[1] - 1, 11)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    distances_k = np.sort(distances[:, k - 1])

    # Grafico k-distance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(distances_k, linewidth=1.5, color='steelblue')
    ax.set_xlabel('Puntos (ordenados por distancia)', fontsize=12)
    ax.set_ylabel(f'Distancia al {k}-esimo vecino', fontsize=12)
    ax.set_title(f'Grafico K-Distance (k={k}) para DBSCAN', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ruta = os.path.join(RESULTS_DIR, '04_kdistance_dbscan.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    # Grid search de eps y min_samples (usando muestra para velocidad)
    np.random.seed(42)
    n_muestra = min(10000, len(X))
    idx_muestra = np.random.choice(len(X), n_muestra, replace=False)
    X_muestra = X[idx_muestra]
    print(f"  Grid search DBSCAN usando muestra de {n_muestra} puntos...")

    eps_values = np.arange(0.5, 1.5, 0.05)
    min_samples_values = [5, 7, 10, 15, 20]

    mejor_score = -1
    mejor_eps = None
    mejor_min_samples = None
    resultados = []

    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X_muestra)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            pct_noise = n_noise / len(labels) * 100

            if n_clusters >= 2 and n_clusters <= 15 and pct_noise < 60:
                mask = labels != -1
                if mask.sum() > n_clusters:
                    score = silhouette_score(X_muestra[mask], labels[mask])
                    resultados.append({
                        'eps': eps, 'min_samples': ms,
                        'n_clusters': n_clusters, 'n_noise': n_noise,
                        'pct_noise': pct_noise, 'silhouette': score
                    })
                    if score > mejor_score:
                        mejor_score = score
                        mejor_eps = eps
                        mejor_min_samples = ms

    lineas = []
    lineas.append("\n  OPTIMIZACION DE DBSCAN")
    lineas.append("=" * 50)

    if mejor_eps is not None:
        lineas.append(f"  Mejor eps: {mejor_eps:.1f}")
        lineas.append(f"  Mejor min_samples: {mejor_min_samples}")
        lineas.append(f"  Mejor Silhouette: {mejor_score:.4f}")

        lineas.append("\n  Top 5 combinaciones:")
        lineas.append("-" * 50)
        df_res = pd.DataFrame(resultados).sort_values('silhouette', ascending=False).head(5)
        lineas.append(df_res.to_string(index=False))
    else:
        # Fallback: usar valores razonables
        mejor_eps = 1.5
        mejor_min_samples = 10
        lineas.append("  No se encontro combinacion optima con grid search.")
        lineas.append(f"  Usando valores por defecto: eps={mejor_eps}, min_samples={mejor_min_samples}")

    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return mejor_eps, mejor_min_samples, texto


def entrenar_dbscan(X, eps, min_samples):
    """Entrena el modelo DBSCAN con los parametros optimizados."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)

    # Guardar modelo
    ruta_modelo = os.path.join(MODELS_DIR, 'dbscan_model.pkl')
    joblib.dump(db, ruta_modelo)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    pct_noise = n_noise / len(labels) * 100

    lineas = []
    lineas.append("\n  MODELO DBSCAN ENTRENADO")
    lineas.append("=" * 50)
    lineas.append(f"  Parametros: eps={eps:.1f}, min_samples={min_samples}")
    lineas.append(f"  Clusters encontrados: {n_clusters}")
    lineas.append(f"  Puntos de ruido: {n_noise} ({pct_noise:.1f}%)")

    # Metricas (excluyendo ruido)
    mask = labels != -1
    if mask.sum() > n_clusters and n_clusters >= 2:
        sil = silhouette_score(X[mask], labels[mask])
        ch = calinski_harabasz_score(X[mask], labels[mask])
        db_index = davies_bouldin_score(X[mask], labels[mask])
        lineas.append(f"  Silhouette Score (sin ruido): {sil:.4f}")
        lineas.append(f"  Calinski-Harabasz Index: {ch:.2f}")
        lineas.append(f"  Davies-Bouldin Index: {db_index:.4f}")

    lineas.append("\n  Tamanio de clusters:")
    lineas.append("-" * 30)
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        pct = count / len(labels) * 100
        nombre = f"Ruido" if cluster == -1 else f"Cluster {cluster}"
        lineas.append(f"    {nombre:12s}: {count:6d} ({pct:.1f}%)")

    lineas.append(f"\n  Modelo guardado en: {ruta_modelo}")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return labels, db, texto


def comparar_modelos(X, labels_kmeans, labels_dbscan):
    """Compara las metricas entre K-means y DBSCAN."""
    lineas = []
    lineas.append("\n  COMPARACION DE MODELOS")
    lineas.append("=" * 50)

    # K-means
    sil_km = silhouette_score(X, labels_kmeans)
    ch_km = calinski_harabasz_score(X, labels_kmeans)
    db_km = davies_bouldin_score(X, labels_kmeans)
    n_clusters_km = len(set(labels_kmeans))

    # DBSCAN
    mask = labels_dbscan != -1
    n_clusters_db = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_noise = (labels_dbscan == -1).sum()

    lineas.append(f"\n  {'Metrica':<30s} {'K-means':>12s} {'DBSCAN':>12s}")
    lineas.append("-" * 56)
    lineas.append(f"  {'Num. clusters':<30s} {n_clusters_km:>12d} {n_clusters_db:>12d}")
    lineas.append(f"  {'Silhouette Score':<30s} {sil_km:>12.4f}", )

    if mask.sum() > n_clusters_db and n_clusters_db >= 2:
        sil_db = silhouette_score(X[mask], labels_dbscan[mask])
        ch_db = calinski_harabasz_score(X[mask], labels_dbscan[mask])
        db_db = davies_bouldin_score(X[mask], labels_dbscan[mask])
        lineas[-1] = f"  {'Silhouette Score':<30s} {sil_km:>12.4f} {sil_db:>12.4f}"
        lineas.append(f"  {'Calinski-Harabasz':<30s} {ch_km:>12.2f} {ch_db:>12.2f}")
        lineas.append(f"  {'Davies-Bouldin':<30s} {db_km:>12.4f} {db_db:>12.4f}")
    else:
        lineas.append(f"  {'Calinski-Harabasz':<30s} {ch_km:>12.2f} {'N/A':>12s}")
        lineas.append(f"  {'Davies-Bouldin':<30s} {db_km:>12.4f} {'N/A':>12s}")

    lineas.append(f"  {'Puntos de ruido':<30s} {'0':>12s} {n_noise:>12d}")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return texto


def guardar_etiquetas(labels_kmeans, labels_dbscan):
    """Guarda las etiquetas de cluster para visualizacion posterior."""
    df_labels = pd.DataFrame({
        'kmeans_cluster': labels_kmeans,
        'dbscan_cluster': labels_dbscan
    })
    ruta = os.path.join(DATA_DIR, 'etiquetas_clusters.csv')
    df_labels.to_csv(ruta, index=False)
    print(f"  Etiquetas guardadas en: {ruta}")


def guardar_reporte(txt_carga, txt_elbow, txt_silhouette, txt_kmeans, txt_dbscan_opt,
                    txt_dbscan, txt_comparacion):
    """Guarda el reporte de implementacion de modelos."""
    reporte = []
    reporte.append("ETAPA 4: IMPLEMENTACION DE MODELOS DE CLUSTERING")
    reporte.append("=" * 50)
    reporte.append(txt_carga)
    reporte.append(txt_elbow)
    reporte.append(txt_silhouette)
    reporte.append(txt_kmeans)
    reporte.append(txt_dbscan_opt)
    reporte.append(txt_dbscan)
    reporte.append(txt_comparacion)

    ruta_reporte = os.path.join(RESULTS_DIR, '04_implementacion_modelos_reporte.txt')
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write("\n".join(reporte))
    print(f"\n  Reporte guardado en: {ruta_reporte}")


if __name__ == '__main__':
    # 1. Cargar features escaladas
    X, txt_carga = cargar_features()
    if X is None:
        exit(1)

    # 2. Metodo del codo
    inertias, txt_elbow = analisis_elbow(X)

    # 3. Analisis de silueta
    mejor_k, scores, txt_silhouette = analisis_silhouette(X)

    # 4. Entrenar K-means
    labels_kmeans, modelo_km, txt_kmeans = entrenar_kmeans(X, mejor_k)

    # 5. Optimizar DBSCAN
    mejor_eps, mejor_ms, txt_dbscan_opt = optimizar_dbscan(X)

    # 6. Entrenar DBSCAN
    labels_dbscan, modelo_db, txt_dbscan = entrenar_dbscan(X, mejor_eps, mejor_ms)

    # 7. Comparar modelos
    txt_comparacion = comparar_modelos(X, labels_kmeans, labels_dbscan)

    # 8. Guardar etiquetas
    guardar_etiquetas(labels_kmeans, labels_dbscan)

    # 9. Guardar reporte
    guardar_reporte(txt_carga, txt_elbow, txt_silhouette, txt_kmeans,
                    txt_dbscan_opt, txt_dbscan, txt_comparacion)
