"""
06 - Visualizacion de Resultados y Conclusiones
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

# Utilidades
import os
import warnings

warnings.filterwarnings('ignore')

# === Rutas del proyecto ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'Data')
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Nombres de features ===
FEATURES_NUM = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping_Cost', 'Aging']


def cargar_todos_los_datos():
    """Carga todos los datos necesarios para visualizacion."""
    archivos = {
        'preprocesados': os.path.join(DATA_DIR, 'datos_preprocesados.csv'),
        'etiquetas': os.path.join(DATA_DIR, 'etiquetas_clusters.csv'),
        'pca': os.path.join(DATA_DIR, 'coordenadas_pca.csv'),
        'tsne': os.path.join(DATA_DIR, 'coordenadas_tsne.csv'),
    }

    for nombre, ruta in archivos.items():
        if not os.path.exists(ruta):
            print(f"  ERROR: No se encontro {ruta}")
            print("  Ejecute primero los scripts anteriores.")
            return None

    df = pd.read_csv(archivos['preprocesados'])
    labels = pd.read_csv(archivos['etiquetas'])
    pca = pd.read_csv(archivos['pca'])
    tsne = pd.read_csv(archivos['tsne'])

    print("=" * 50)
    print("  DATOS CARGADOS PARA VISUALIZACION")
    print("=" * 50)
    print(f"  Dataset: {df.shape}")
    print(f"  Etiquetas: {labels.shape}")
    print(f"  PCA: {pca.shape}")
    print(f"  t-SNE: {tsne.shape}")
    print("=" * 50)

    return df, labels, pca, tsne


def visualizar_pca_por_clusters(pca, labels):
    """Genera scatter plots PCA coloreados por cluster de cada modelo."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # K-means
    scatter1 = ax1.scatter(pca['PC1'], pca['PC2'],
                           c=labels['kmeans_cluster'], cmap='tab10',
                           s=3, alpha=0.5)
    ax1.set_xlabel('PC1', fontsize=12)
    ax1.set_ylabel('PC2', fontsize=12)
    ax1.set_title('PCA - Clusters K-means', fontsize=14, fontweight='bold')
    plt.colorbar(scatter1, ax=ax1, label='Cluster')

    # DBSCAN
    colores_dbscan = labels['dbscan_cluster'].copy()
    mask_ruido = colores_dbscan == -1

    ax2.scatter(pca['PC1'][mask_ruido], pca['PC2'][mask_ruido],
                c='lightgray', s=3, alpha=0.3, label='Ruido')
    if (~mask_ruido).any():
        scatter2 = ax2.scatter(pca['PC1'][~mask_ruido], pca['PC2'][~mask_ruido],
                               c=colores_dbscan[~mask_ruido], cmap='tab10',
                               s=3, alpha=0.5)
        plt.colorbar(scatter2, ax=ax2, label='Cluster')
    ax2.set_xlabel('PC1', fontsize=12)
    ax2.set_ylabel('PC2', fontsize=12)
    ax2.set_title('PCA - Clusters DBSCAN', fontsize=14, fontweight='bold')
    ax2.legend(markerscale=5, fontsize=10)

    plt.tight_layout()
    ruta = os.path.join(RESULTS_DIR, '06_pca_clusters_comparacion.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")


def visualizar_tsne_por_clusters(tsne, labels):
    """Genera scatter plots t-SNE coloreados por cluster de cada modelo."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # K-means
    scatter1 = ax1.scatter(tsne['tSNE1'], tsne['tSNE2'],
                           c=labels['kmeans_cluster'], cmap='tab10',
                           s=3, alpha=0.5)
    ax1.set_xlabel('t-SNE 1', fontsize=12)
    ax1.set_ylabel('t-SNE 2', fontsize=12)
    ax1.set_title('t-SNE - Clusters K-means', fontsize=14, fontweight='bold')
    plt.colorbar(scatter1, ax=ax1, label='Cluster')

    # DBSCAN
    colores_dbscan = labels['dbscan_cluster'].copy()
    mask_ruido = colores_dbscan == -1

    ax2.scatter(tsne['tSNE1'][mask_ruido], tsne['tSNE2'][mask_ruido],
                c='lightgray', s=3, alpha=0.3, label='Ruido')
    if (~mask_ruido).any():
        scatter2 = ax2.scatter(tsne['tSNE1'][~mask_ruido], tsne['tSNE2'][~mask_ruido],
                               c=colores_dbscan[~mask_ruido], cmap='tab10',
                               s=3, alpha=0.5)
        plt.colorbar(scatter2, ax=ax2, label='Cluster')
    ax2.set_xlabel('t-SNE 1', fontsize=12)
    ax2.set_ylabel('t-SNE 2', fontsize=12)
    ax2.set_title('t-SNE - Clusters DBSCAN', fontsize=14, fontweight='bold')
    ax2.legend(markerscale=5, fontsize=10)

    plt.tight_layout()
    ruta = os.path.join(RESULTS_DIR, '06_tsne_clusters_comparacion.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")


def visualizar_comparacion_4panel(pca, tsne, labels):
    """Genera panel 2x2: PCA/t-SNE x K-means/DBSCAN."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Comparacion de Modelos y Metodos de Reduccion', fontsize=16, fontweight='bold')

    mask_ruido = labels['dbscan_cluster'] == -1

    # (0,0) K-means + PCA
    axes[0, 0].scatter(pca['PC1'], pca['PC2'],
                       c=labels['kmeans_cluster'], cmap='tab10', s=3, alpha=0.5)
    axes[0, 0].set_title('K-means + PCA', fontsize=13)
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')

    # (0,1) DBSCAN + PCA
    axes[0, 1].scatter(pca['PC1'][mask_ruido], pca['PC2'][mask_ruido],
                       c='lightgray', s=3, alpha=0.3, label='Ruido')
    if (~mask_ruido).any():
        axes[0, 1].scatter(pca['PC1'][~mask_ruido], pca['PC2'][~mask_ruido],
                           c=labels['dbscan_cluster'][~mask_ruido], cmap='tab10', s=3, alpha=0.5)
    axes[0, 1].set_title('DBSCAN + PCA', fontsize=13)
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    axes[0, 1].legend(markerscale=5, fontsize=9)

    # (1,0) K-means + t-SNE
    axes[1, 0].scatter(tsne['tSNE1'], tsne['tSNE2'],
                       c=labels['kmeans_cluster'], cmap='tab10', s=3, alpha=0.5)
    axes[1, 0].set_title('K-means + t-SNE', fontsize=13)
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')

    # (1,1) DBSCAN + t-SNE
    axes[1, 1].scatter(tsne['tSNE1'][mask_ruido], tsne['tSNE2'][mask_ruido],
                       c='lightgray', s=3, alpha=0.3, label='Ruido')
    if (~mask_ruido).any():
        axes[1, 1].scatter(tsne['tSNE1'][~mask_ruido], tsne['tSNE2'][~mask_ruido],
                           c=labels['dbscan_cluster'][~mask_ruido], cmap='tab10', s=3, alpha=0.5)
    axes[1, 1].set_title('DBSCAN + t-SNE', fontsize=13)
    axes[1, 1].set_xlabel('t-SNE 1')
    axes[1, 1].set_ylabel('t-SNE 2')
    axes[1, 1].legend(markerscale=5, fontsize=9)

    plt.tight_layout()
    ruta = os.path.join(RESULTS_DIR, '06_comparacion_modelos_4panel.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")


def generar_tabla_perfiles(df, labels):
    """Genera tablas de caracteristicas medias por cluster."""
    textos = []

    for modelo, col_label in [('kmeans', 'kmeans_cluster'), ('dbscan', 'dbscan_cluster')]:
        df_temp = df[FEATURES_NUM].copy()
        df_temp['Cluster'] = labels[col_label].values

        # Tabla de medias por cluster
        tabla = df_temp.groupby('Cluster')[FEATURES_NUM].mean().round(2)
        tabla['Count'] = df_temp.groupby('Cluster').size()
        tabla['Pct'] = (tabla['Count'] / len(df) * 100).round(1)

        # Guardar CSV
        ruta_csv = os.path.join(RESULTS_DIR, f'06_tabla_perfiles_{modelo}.csv')
        tabla.to_csv(ruta_csv)
        print(f"  Guardado: {ruta_csv}")

        # Heatmap de perfiles
        fig, ax = plt.subplots(figsize=(12, max(4, len(tabla) * 0.8)))
        tabla_norm = tabla[FEATURES_NUM].copy()
        # Normalizar por columna para mejor visualizacion
        for col in FEATURES_NUM:
            col_min = tabla_norm[col].min()
            col_max = tabla_norm[col].max()
            if col_max > col_min:
                tabla_norm[col] = (tabla_norm[col] - col_min) / (col_max - col_min)

        sns.heatmap(tabla_norm, annot=tabla[FEATURES_NUM].values, fmt='.1f',
                    cmap='YlOrRd', linewidths=0.5, ax=ax)
        nombre_modelo = 'K-means' if modelo == 'kmeans' else 'DBSCAN'
        ax.set_title(f'Perfiles de Cluster - {nombre_modelo}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cluster', fontsize=12)

        plt.tight_layout()
        ruta_hm = os.path.join(RESULTS_DIR, f'06_heatmap_perfiles_{modelo}.png')
        plt.savefig(ruta_hm, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Guardado: {ruta_hm}")

        # Texto para reporte
        textos.append(f"\n  PERFILES DE CLUSTER - {nombre_modelo.upper()}")
        textos.append("=" * 50)
        textos.append(tabla.to_string())
        textos.append("=" * 50)

    return "\n".join(textos)


def visualizar_distribucion_clusters(labels):
    """Grafico de barras comparando tamanio de clusters."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # K-means
    km_counts = pd.Series(labels['kmeans_cluster']).value_counts().sort_index()
    ax1.bar(km_counts.index.astype(str), km_counts.values, color='steelblue', edgecolor='white')
    ax1.set_xlabel('Cluster', fontsize=12)
    ax1.set_ylabel('Cantidad', fontsize=12)
    ax1.set_title('Distribucion de Clusters - K-means', fontsize=13, fontweight='bold')
    for i, v in enumerate(km_counts.values):
        ax1.text(i, v + len(labels) * 0.005, str(v), ha='center', fontsize=9)

    # DBSCAN
    db_counts = pd.Series(labels['dbscan_cluster']).value_counts().sort_index()
    colores = ['lightgray' if idx == -1 else 'coral' for idx in db_counts.index]
    etiquetas = ['Ruido' if idx == -1 else f'C{idx}' for idx in db_counts.index]
    ax2.bar(etiquetas, db_counts.values, color=colores, edgecolor='white')
    ax2.set_xlabel('Cluster', fontsize=12)
    ax2.set_ylabel('Cantidad', fontsize=12)
    ax2.set_title('Distribucion de Clusters - DBSCAN', fontsize=13, fontweight='bold')
    for i, v in enumerate(db_counts.values):
        ax2.text(i, v + len(labels) * 0.005, str(v), ha='center', fontsize=9)

    plt.tight_layout()
    ruta = os.path.join(RESULTS_DIR, '06_distribucion_clusters.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")


def visualizar_boxplots_por_cluster(df, labels):
    """Boxplots de features principales separados por cluster K-means."""
    df_temp = df[FEATURES_NUM].copy()
    df_temp['Cluster'] = labels['kmeans_cluster'].values

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Distribucion de Features por Cluster (K-means)', fontsize=16, fontweight='bold')

    for i, col in enumerate(FEATURES_NUM):
        ax = axes[i // 3, i % 3]
        sns.boxplot(x='Cluster', y=col, data=df_temp, ax=ax, palette='Set2')
        ax.set_title(col, fontsize=12)

    plt.tight_layout()
    ruta = os.path.join(RESULTS_DIR, '06_boxplots_por_cluster_kmeans.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")


def generar_reflexion(df, labels):
    """Genera la reflexion y conclusiones del analisis."""
    lineas = []

    # --- Pregunta 1: Tipos de perfiles ---
    lineas.append("\n  REFLEXION Y CONCLUSIONES")
    lineas.append("=" * 50)
    lineas.append("\n  1. TIPOS DE PERFILES IDENTIFICADOS")
    lineas.append("-" * 50)

    df_temp = df[FEATURES_NUM].copy()
    df_temp['Cluster'] = labels['kmeans_cluster'].values
    medias = df_temp.groupby('Cluster')[FEATURES_NUM].mean()
    medias_global = df[FEATURES_NUM].mean()

    for cluster in medias.index:
        lineas.append(f"\n  Cluster {cluster}:")
        caracteristicas = []
        for feat in FEATURES_NUM:
            val = medias.loc[cluster, feat]
            media = medias_global[feat]
            if val > media * 1.15:
                caracteristicas.append(f"{feat} alto ({val:.1f} vs media {media:.1f})")
            elif val < media * 0.85:
                caracteristicas.append(f"{feat} bajo ({val:.1f} vs media {media:.1f})")
        if caracteristicas:
            for c in caracteristicas:
                lineas.append(f"    - {c}")
        else:
            lineas.append(f"    - Perfil cercano al promedio general")

    # --- Pregunta 2: Diferencias entre modelos ---
    lineas.append("\n\n  2. DIFERENCIAS CLAVE ENTRE K-MEANS Y DBSCAN")
    lineas.append("-" * 50)

    n_km = len(set(labels['kmeans_cluster']))
    n_db = len(set(labels['dbscan_cluster'])) - (1 if -1 in labels['dbscan_cluster'].values else 0)
    n_ruido = (labels['dbscan_cluster'] == -1).sum()
    pct_ruido = n_ruido / len(labels) * 100

    lineas.append(f"  - K-means identifico {n_km} clusters, asignando TODOS los puntos a un cluster.")
    lineas.append(f"  - DBSCAN identifico {n_db} clusters, con {n_ruido} puntos de ruido ({pct_ruido:.1f}%).")
    lineas.append(f"  - K-means asume clusters esfericos de tamanio similar.")
    lineas.append(f"  - DBSCAN detecta clusters de forma arbitraria y densidad variable.")
    lineas.append(f"  - K-means requiere especificar K a priori; DBSCAN lo determina automaticamente.")
    lineas.append(f"  - DBSCAN es mas robusto a outliers al clasificarlos como ruido.")

    # --- Pregunta 3: Limitaciones ---
    lineas.append("\n\n  3. LIMITACIONES ENCONTRADAS Y PROPUESTAS DE SOLUCION")
    lineas.append("-" * 50)

    lineas.append("  a) Datos a nivel de transaccion, no de cliente:")
    lineas.append("     - Cada fila es una orden, no un cliente unico.")
    lineas.append("     - Solucion: agregar datos por Customer_Id antes de clustering.")

    lineas.append("\n  b) Variables temporales no aprovechadas:")
    lineas.append("     - Order_Date y Time se eliminaron, pero podrian revelar patrones estacionales.")
    lineas.append("     - Solucion: crear features derivadas (mes, dia de semana, hora).")

    lineas.append("\n  c) Maldicion de la dimensionalidad en DBSCAN:")
    lineas.append("     - Con 6 dimensiones, la nocion de densidad se diluye.")
    lineas.append("     - Solucion: aplicar PCA antes de DBSCAN para reducir dimensiones.")

    lineas.append("\n  d) Sensibilidad de DBSCAN a sus parametros:")
    lineas.append("     - Pequenos cambios en eps generan resultados muy diferentes.")
    lineas.append("     - Solucion: usar OPTICS como alternativa mas robusta.")

    lineas.append("\n  e) Ausencia de validacion externa:")
    lineas.append("     - No existe ground truth para evaluar la calidad real de los clusters.")
    lineas.append("     - Solucion: validar con expertos del dominio de negocio.")

    lineas.append("\n" + "=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return texto


def guardar_reporte(txt_perfiles, txt_reflexion):
    """Guarda el reporte de visualizacion y conclusiones."""
    reporte = []
    reporte.append("ETAPA 6: VISUALIZACION DE RESULTADOS Y CONCLUSIONES")
    reporte.append("=" * 50)
    reporte.append(txt_perfiles)
    reporte.append(txt_reflexion)

    ruta_reporte = os.path.join(RESULTS_DIR, '06_visualizacion_conclusiones_reporte.txt')
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write("\n".join(reporte))
    print(f"\n  Reporte guardado en: {ruta_reporte}")


if __name__ == '__main__':
    # 1. Cargar todos los datos
    resultado = cargar_todos_los_datos()
    if resultado is None:
        exit(1)
    df, labels, pca, tsne = resultado

    # 2. Visualizaciones PCA por clusters
    visualizar_pca_por_clusters(pca, labels)

    # 3. Visualizaciones t-SNE por clusters
    visualizar_tsne_por_clusters(tsne, labels)

    # 4. Comparacion 4 paneles
    visualizar_comparacion_4panel(pca, tsne, labels)

    # 5. Tablas de perfiles
    txt_perfiles = generar_tabla_perfiles(df, labels)

    # 6. Distribucion de clusters
    visualizar_distribucion_clusters(labels)

    # 7. Boxplots por cluster
    visualizar_boxplots_por_cluster(df, labels)

    # 8. Reflexion y conclusiones
    txt_reflexion = generar_reflexion(df, labels)

    # 9. Guardar reporte
    guardar_reporte(txt_perfiles, txt_reflexion)
