"""
02 - Carga de Datos y Analisis Exploratorio (EDA)
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


def cargar_datos():
    """Carga el dataset E-commerce desde Data/."""
    ruta_datos = os.path.join(DATA_DIR, 'E-commerce Dataset.csv')

    if not os.path.exists(ruta_datos):
        print(f"  ERROR: No se encontro el archivo en {ruta_datos}")
        return None, ""

    df = pd.read_csv(ruta_datos)

    lineas = []
    lineas.append("=" * 50)
    lineas.append("  CARGA DEL DATASET")
    lineas.append("=" * 50)
    lineas.append(f"  Filas:    {df.shape[0]}")
    lineas.append(f"  Columnas: {df.shape[1]}")
    lineas.append(f"  Nombres:  {list(df.columns)}")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return df, texto


def analisis_valores_nulos(df):
    """Analiza y trata los valores nulos del dataset."""
    lineas = []
    lineas.append("\n  ANALISIS DE VALORES NULOS")
    lineas.append("=" * 50)

    nulos_por_columna = df.isnull().sum()
    total_nulos = nulos_por_columna.sum()

    for col in df.columns:
        n = nulos_por_columna[col]
        if n > 0:
            lineas.append(f"  {col:30s} -> {n} nulos")

    lineas.append(f"\n  Total valores nulos: {total_nulos}")
    lineas.append(f"  Porcentaje del dataset: {total_nulos / len(df) * 100:.4f}%")

    # Eliminar filas con nulos (son muy pocas)
    filas_antes = len(df)
    df_limpio = df.dropna()
    filas_despues = len(df_limpio)
    filas_eliminadas = filas_antes - filas_despues

    lineas.append(f"\n  Estrategia: Eliminacion de filas con nulos")
    lineas.append(f"  Filas eliminadas: {filas_eliminadas}")
    lineas.append(f"  Filas restantes:  {filas_despues}")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return df_limpio, texto


def analisis_estadistico(df):
    """Genera estadisticas descriptivas para variables numericas y categoricas."""
    lineas = []

    # Estadisticas numericas
    lineas.append("\n  ESTADISTICAS DESCRIPTIVAS - VARIABLES NUMERICAS")
    lineas.append("=" * 50)
    numericas = df.select_dtypes(include=[np.number])
    lineas.append(numericas.describe().to_string())

    # Columnas de alta cardinalidad que seran eliminadas (resumen breve)
    columnas_alta_cardinalidad = ['Order_Date', 'Time', 'Customer_Id', 'Product', 'Product_Category']

    lineas.append("\n\n  VARIABLES DE ALTA CARDINALIDAD (se eliminaran en preprocesamiento)")
    lineas.append("=" * 50)
    for col in columnas_alta_cardinalidad:
        if col in df.columns:
            n_unicos = df[col].nunique()
            n_nulos = df[col].isnull().sum()
            top_valor = df[col].value_counts().index[0]
            top_freq = df[col].value_counts().iloc[0]
            top_pct = top_freq / len(df) * 100
            lineas.append(f"\n  {col}:")
            lineas.append("-" * 30)
            lineas.append(f"    Valores unicos: {n_unicos}")
            lineas.append(f"    Valores nulos:  {n_nulos}")
            lineas.append(f"    Valor mas frecuente: {top_valor} -> {top_freq} ({top_pct:.1f}%)")
            if col == 'Order_Date':
                lineas.append(f"    Rango: {df[col].min()} a {df[col].max()}")
                lineas.append(f"    Motivo de exclusion: variable temporal, no aporta a segmentacion de perfiles")
            elif col == 'Time':
                lineas.append(f"    Motivo de exclusion: variable temporal, no aporta a segmentacion de perfiles")
            elif col == 'Customer_Id':
                lineas.append(f"    Motivo de exclusion: identificador unico, no es una feature")
            elif col in ['Product', 'Product_Category']:
                lineas.append(f"    Motivo de exclusion: alta cardinalidad ({n_unicos} valores), distorsiona clustering")

    # Estadisticas categoricas (solo las de baja cardinalidad que se conservan)
    lineas.append("\n\n  ESTADISTICAS DESCRIPTIVAS - VARIABLES CATEGORICAS")
    lineas.append("=" * 50)
    categoricas = df.select_dtypes(include=['object'])
    for col in categoricas.columns:
        if col in columnas_alta_cardinalidad:
            continue
        lineas.append(f"\n  {col}:")
        lineas.append("-" * 30)
        conteo = df[col].value_counts()
        for val, cnt in conteo.items():
            pct = cnt / len(df) * 100
            lineas.append(f"    {val:30s} -> {cnt:6d} ({pct:.1f}%)")

    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return texto


def visualizar_distribuciones(df):
    """Genera histogramas y boxplots de variables numericas."""
    features_num = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping_Cost', 'Aging']

    # Histogramas con KDE
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Distribuciones de Variables Numericas', fontsize=16, fontweight='bold')

    for i, col in enumerate(features_num):
        ax = axes[i // 3, i % 3]
        sns.histplot(df[col], kde=True, ax=ax, color='steelblue', edgecolor='white')
        ax.set_title(col, fontsize=12)
        ax.set_xlabel('')

    plt.tight_layout()
    ruta = os.path.join(RESULTS_DIR, '02_distribuciones_numericas.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    # Boxplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Boxplots de Variables Numericas', fontsize=16, fontweight='bold')

    for i, col in enumerate(features_num):
        ax = axes[i // 3, i % 3]
        sns.boxplot(y=df[col], ax=ax, color='lightcoral')
        ax.set_title(col, fontsize=12)

    plt.tight_layout()
    ruta = os.path.join(RESULTS_DIR, '02_boxplots_numericas.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")


def visualizar_categoricas(df):
    """Genera graficos de barras para variables categoricas."""
    categoricas = ['Gender', 'Device_Type', 'Customer_Login_type', 'Order_Priority', 'Payment_method']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Distribuciones de Variables Categoricas', fontsize=16, fontweight='bold')

    for i, col in enumerate(categoricas):
        ax = axes[i // 3, i % 3]
        conteo = df[col].value_counts()
        sns.barplot(x=conteo.index, y=conteo.values, ax=ax, palette='Set2')
        ax.set_title(col, fontsize=12)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

    # Ocultar el subplot vacio (2x3 = 6, solo hay 5 categoricas)
    axes[1, 2].set_visible(False)

    plt.tight_layout()
    ruta = os.path.join(RESULTS_DIR, '02_distribuciones_categoricas.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")


def analisis_correlacion(df):
    """Genera la matriz de correlacion para variables numericas."""
    features_num = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping_Cost', 'Aging']
    corr = df[features_num].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Matriz de Correlacion - Variables Numericas', fontsize=14, fontweight='bold')

    plt.tight_layout()
    ruta = os.path.join(RESULTS_DIR, '02_matriz_correlacion.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    lineas = []
    lineas.append("\n  MATRIZ DE CORRELACION")
    lineas.append("=" * 50)
    lineas.append(corr.to_string())

    # Correlaciones altas
    lineas.append("\n  CORRELACIONES DESTACADAS (|r| > 0.5):")
    lineas.append("-" * 50)
    for i in range(len(features_num)):
        for j in range(i + 1, len(features_num)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                lineas.append(f"  {features_num[i]} vs {features_num[j]}: r = {r:.3f}")

    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return texto


def identificar_columnas_irrelevantes(df):
    """Identifica y documenta las columnas a eliminar o transformar."""
    columnas_eliminar = ['Order_Date', 'Time', 'Customer_Id', 'Product', 'Product_Category', 'Device_Type']

    lineas = []
    lineas.append("\n  IDENTIFICACION DE COLUMNAS IRRELEVANTES")
    lineas.append("=" * 50)
    lineas.append("  Columnas a eliminar para clustering:")
    lineas.append(f"    - Order_Date: variable temporal, no aporta a segmentacion de perfiles")
    lineas.append(f"    - Time: variable temporal, no aporta a segmentacion de perfiles")
    lineas.append(f"    - Customer_Id: identificador unico, no es una feature")
    lineas.append(f"    - Product: alta cardinalidad ({df['Product'].nunique()} valores unicos)")
    lineas.append(f"    - Product_Category: alta cardinalidad ({df['Product_Category'].nunique()} valores unicos)")
    lineas.append(f"    - Device_Type: baja variabilidad ({df['Device_Type'].nunique()} valores unicos)")

    lineas.append("\n  Columnas que se conservan:")
    columnas_conservar = [c for c in df.columns if c not in columnas_eliminar]
    for col in columnas_conservar:
        lineas.append(f"    - {col}")

    df_reducido = df.drop(columns=columnas_eliminar)

    lineas.append(f"\n  Dataset reducido: {df_reducido.shape[0]} filas x {df_reducido.shape[1]} columnas")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return df_reducido, texto


def guardar_datos_limpios(df):
    """Guarda el dataset limpio para las siguientes fases."""
    ruta = os.path.join(DATA_DIR, 'E-commerce_limpio.csv')
    df.to_csv(ruta, index=False)
    print(f"\n  Dataset limpio guardado en: {ruta}")
    print(f"  Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")


def guardar_reporte(txt_carga, txt_nulos, txt_estadistico, txt_correlacion, txt_columnas):
    """Guarda el reporte completo del analisis exploratorio."""
    reporte = []
    reporte.append("ETAPA 2: CARGA DE DATOS Y ANALISIS EXPLORATORIO (EDA)")
    reporte.append("=" * 50)
    reporte.append(txt_carga)
    reporte.append(txt_nulos)
    reporte.append(txt_estadistico)
    reporte.append(txt_correlacion)
    reporte.append(txt_columnas)

    ruta_reporte = os.path.join(RESULTS_DIR, '02_analisis_exploratorio_reporte.txt')
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write("\n".join(reporte))
    print(f"\n  Reporte guardado en: {ruta_reporte}")


if __name__ == '__main__':
    # 1. Cargar datos
    df, txt_carga = cargar_datos()
    if df is None:
        exit(1)

    # 2. Analisis de valores nulos y limpieza
    df, txt_nulos = analisis_valores_nulos(df)

    # 3. Analisis estadistico
    txt_estadistico = analisis_estadistico(df)

    # 4. Visualizaciones de distribuciones
    visualizar_distribuciones(df)

    # 5. Visualizaciones de categoricas
    visualizar_categoricas(df)

    # 6. Analisis de correlacion
    txt_correlacion = analisis_correlacion(df)

    # 7. Identificar y eliminar columnas irrelevantes
    df_limpio, txt_columnas = identificar_columnas_irrelevantes(df)

    # 8. Guardar dataset limpio
    guardar_datos_limpios(df_limpio)

    # 9. Guardar reporte
    guardar_reporte(txt_carga, txt_nulos, txt_estadistico, txt_correlacion, txt_columnas)
