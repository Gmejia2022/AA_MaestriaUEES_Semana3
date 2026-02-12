"""
05 - Reduccion de Dimensionalidad (PCA y t-SNE)
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

# Reduccion de dimensionalidad
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Persistencia de modelos
import joblib

# Utilidades
import os
import time
import warnings

warnings.filterwarnings('ignore')

# === Rutas del proyecto ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'Data')
MODELS_DIR = os.path.join(BASE_DIR, 'Models')
os.makedirs(RESULTS_DIR, exist_ok=True)


def cargar_datos():
    """Carga features escaladas y etiquetas de clusters."""
    ruta_features = os.path.join(DATA_DIR, 'features_scaled.csv')
    ruta_labels = os.path.join(DATA_DIR, 'etiquetas_clusters.csv')

    for ruta in [ruta_features, ruta_labels]:
        if not os.path.exists(ruta):
            print(f"  ERROR: No se encontro {ruta}")
            print("  Ejecute primero los scripts anteriores.")
            return None, None, ""

    X = pd.read_csv(ruta_features).values
    labels = pd.read_csv(ruta_labels)

    lineas = []
    lineas.append("=" * 50)
    lineas.append("  CARGA DE DATOS PARA REDUCCION DE DIMENSIONALIDAD")
    lineas.append("=" * 50)
    lineas.append(f"  Features: {X.shape}")
    lineas.append(f"  Etiquetas cargadas: kmeans_cluster, dbscan_cluster")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return X, labels, texto


def aplicar_pca(X):
    """Aplica PCA para reduccion lineal a 2 componentes."""
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # Guardar modelo
    ruta_modelo = os.path.join(MODELS_DIR, 'pca_model.pkl')
    joblib.dump(pca, ruta_modelo)

    lineas = []
    lineas.append("\n  PCA - REDUCCION A 2 COMPONENTES")
    lineas.append("=" * 50)
    lineas.append(f"  Varianza explicada PC1: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    lineas.append(f"  Varianza explicada PC2: {pca.explained_variance_ratio_[1]:.4f} ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    lineas.append(f"  Varianza acumulada:     {sum(pca.explained_variance_ratio_):.4f} ({sum(pca.explained_variance_ratio_)*100:.2f}%)")
    lineas.append(f"\n  Modelo guardado en: {ruta_modelo}")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return X_pca, pca, texto


def analisis_varianza_pca(X):
    """Analiza la varianza explicada por cada componente PCA."""
    n_components = min(X.shape[1], 6)
    pca_full = PCA(n_components=n_components, random_state=42)
    pca_full.fit(X)

    varianza = pca_full.explained_variance_ratio_
    varianza_acum = np.cumsum(varianza)

    # Grafico de varianza explicada
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Scree plot
    componentes = range(1, n_components + 1)
    ax1.bar(componentes, varianza, color='steelblue', alpha=0.8, edgecolor='white')
    ax1.set_xlabel('Componente Principal', fontsize=12)
    ax1.set_ylabel('Varianza Explicada', fontsize=12)
    ax1.set_title('Varianza por Componente (Scree Plot)', fontsize=13, fontweight='bold')
    ax1.set_xticks(list(componentes))

    # Varianza acumulada
    ax2.plot(componentes, varianza_acum, 'ro-', linewidth=2, markersize=8)
    ax2.axhline(y=0.80, color='gray', linestyle='--', alpha=0.5, label='80%')
    ax2.axhline(y=0.90, color='gray', linestyle=':', alpha=0.5, label='90%')
    ax2.set_xlabel('Numero de Componentes', fontsize=12)
    ax2.set_ylabel('Varianza Acumulada', fontsize=12)
    ax2.set_title('Varianza Acumulada', fontsize=13, fontweight='bold')
    ax2.set_xticks(list(componentes))
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    ruta = os.path.join(RESULTS_DIR, '05_pca_varianza_explicada.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    lineas = []
    lineas.append("\n  ANALISIS DE VARIANZA PCA (TODOS LOS COMPONENTES)")
    lineas.append("=" * 50)
    for i in range(n_components):
        lineas.append(f"  PC{i+1}: {varianza[i]:.4f} ({varianza[i]*100:.2f}%)  |  Acumulada: {varianza_acum[i]:.4f} ({varianza_acum[i]*100:.2f}%)")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return texto


def aplicar_tsne(X):
    """Aplica t-SNE para reduccion no lineal a 2 componentes."""
    print("\n  Ejecutando t-SNE (esto puede tomar varios minutos)...")
    inicio = time.time()

    tsne = TSNE(n_components=2, random_state=42, perplexity=30,
                max_iter=1000, learning_rate='auto')
    X_tsne = tsne.fit_transform(X)

    tiempo = time.time() - inicio

    lineas = []
    lineas.append("\n  t-SNE - REDUCCION A 2 COMPONENTES")
    lineas.append("=" * 50)
    lineas.append(f"  Parametros: perplexity=30, max_iter=1000, learning_rate=auto")
    lineas.append(f"  Tiempo de ejecucion: {tiempo:.1f} segundos")
    lineas.append(f"  KL Divergence: {tsne.kl_divergence_:.4f}")
    lineas.append(f"  Dimensiones resultado: {X_tsne.shape}")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return X_tsne, texto


def guardar_coordenadas(X_pca, X_tsne):
    """Guarda las coordenadas reducidas para visualizacion."""
    # PCA
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    ruta_pca = os.path.join(DATA_DIR, 'coordenadas_pca.csv')
    df_pca.to_csv(ruta_pca, index=False)
    print(f"  Coordenadas PCA guardadas en: {ruta_pca}")

    # t-SNE
    df_tsne = pd.DataFrame(X_tsne, columns=['tSNE1', 'tSNE2'])
    ruta_tsne = os.path.join(DATA_DIR, 'coordenadas_tsne.csv')
    df_tsne.to_csv(ruta_tsne, index=False)
    print(f"  Coordenadas t-SNE guardadas en: {ruta_tsne}")


def guardar_reporte(txt_carga, txt_pca, txt_varianza, txt_tsne):
    """Guarda el reporte de reduccion de dimensionalidad."""
    reporte = []
    reporte.append("ETAPA 5: REDUCCION DE DIMENSIONALIDAD")
    reporte.append("=" * 50)
    reporte.append(txt_carga)
    reporte.append(txt_pca)
    reporte.append(txt_varianza)
    reporte.append(txt_tsne)

    ruta_reporte = os.path.join(RESULTS_DIR, '05_reduccion_dimensionalidad_reporte.txt')
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write("\n".join(reporte))
    print(f"\n  Reporte guardado en: {ruta_reporte}")


if __name__ == '__main__':
    # 1. Cargar datos
    X, labels, txt_carga = cargar_datos()
    if X is None:
        exit(1)

    # 2. Aplicar PCA
    X_pca, pca, txt_pca = aplicar_pca(X)

    # 3. Analisis de varianza PCA
    txt_varianza = analisis_varianza_pca(X)

    # 4. Aplicar t-SNE
    X_tsne, txt_tsne = aplicar_tsne(X)

    # 5. Guardar coordenadas
    guardar_coordenadas(X_pca, X_tsne)

    # 6. Guardar reporte
    guardar_reporte(txt_carga, txt_pca, txt_varianza, txt_tsne)
