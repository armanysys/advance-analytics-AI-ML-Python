# Establece el tamaño de tu Data Set y su variable objetivo

## Entendiendo tu DataSet dentro de Aprendizaje Supervisado

En el aprendizaje supervisado, nuestro objetivo es mapear las características de entrada a una variable objetivo. Antes de construir cualquier modelo, es esencial explorar y comprender el conjunto de datos a fondo. Este capítulo te guiará a través de los pasos clave en este proceso, que incluyen:

- **Comprender la Variable Objetivo**
- **Explorar el Espacio de Características**
- **Tamaño del DataSet para Modelar**

## 1. Comprender la Variable Objetivo

La variable objetivo es el valor que tu modelo intenta predecir. Su tipo determina el tipo de modelo que utilizarás. Comprender su distribución es clave para asegurar que los datos sean adecuados para la tarea.

Aquí tienes tu contenido en formato Markdown:


### 1.1 Ejemplo de Clasificación Binaria

En una tarea de clasificación binaria, la variable objetivo solo puede tener dos valores (por ejemplo, `0` y `1`). Es importante verificar si el conjunto de datos está equilibrado. Si las clases están desbalanceadas, es posible que necesites técnicas como el remuestreo.

#### Código Ejemplo: Graficar la Distribución de la Variable Binaria

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Cargar el conjunto de datos y convertir a DataFrame
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Graficar la distribución de la variable objetivo
sns.countplot(x='target', data=df, palette='Set2')
plt.title('Distribución de la Variable Objetivo (Binaria)')
plt.xlabel('Clase Objetivo')
plt.ylabel('Cantidad')
plt.show()
```

#### Explicación

En este ejemplo, utilizamos el conjunto de datos de cáncer de mama, donde la variable objetivo es binaria (`0` para benigno, `1` para maligno). El gráfico de barras muestra la distribución de las clases.

### 1.2 Ejemplo de Clasificación Multiclase

En una tarea de clasificación multiclase, la variable objetivo puede tener más de dos categorías. Comprender la distribución de estas categorías es importante, especialmente si algunas clases están subrepresentadas.

##### Código Ejemplo: Graficar la Distribución de la Variable Multiclase

```python
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos de iris y convertir a DataFrame
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['target'] = iris.target

# Graficar la distribución de la variable objetivo
sns.countplot(x='target', data=df_iris, palette='Set3')
plt.title('Distribución de la Variable Objetivo (Multiclase)')
plt.xlabel('Clase Objetivo')
plt.ylabel('Cantidad')
plt.show()
```

#### Explicación
En el conjunto de datos de Iris, la variable objetivo tiene tres clases (`0`, `1`, `2`). El gráfico de barras muestra la distribución de cada clase.

Aquí tienes tu contenido en formato Markdown:


### 1.3 Ejemplo de la Variable Continua

Cuando la variable objetivo es continua, estamos hablando de modelaje predictivo de regresión. Visualizar su distribución ayuda a entender el rango, la distribución y posibles valores atípicos.

#### Código Ejemplo: Graficar la Distribución de la Variable Objetivo Continua

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el conjunto de datos y convertir a DataFrame
california = fetch_california_housing()
df_california = pd.DataFrame(california.data, columns=california.feature_names)
df_california['target'] = california.target

# Graficar histograma para la variable objetivo
plt.figure(figsize=(10, 6))
sns.histplot(df_california['target'], kde=True)
plt.title('Distribución de la Variable Objetivo (Regresión)')
plt.xlabel('Valor Objetivo')
plt.ylabel('Frecuencia')
plt.show()

# Graficar boxplot para identificar valores atípicos
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_california['target'])
plt.title('Boxplot de la Variable Objetivo')
plt.xlabel('Valor Objetivo')
plt.show()
```

#### Explicación

Aquí utilizamos el conjunto de datos de viviendas de California para graficar un histograma y un diagrama de caja de la variable objetivo (precios de viviendas), lo que permite identificar la distribución y posibles valores atípicos.


### 2. Explorando el Espacio de Características

El espacio de características consiste en las variables de entrada o predictores del conjunto de datos. Debes explorar el espacio de características para detectar valores faltantes, correlaciones y relaciones con la variable objetivo.

#### 2.1 Verificar Valores Faltantes

Los valores faltantes pueden afectar el rendimiento de los modelos de aprendizaje automático. Una forma rápida de verificarlos es contando los valores faltantes por característica.

**Código Ejemplo: Verificar Valores Faltantes**

```python
# Verificar valores faltantes
missing_values = df.isnull().sum()
print("Valores faltantes por columna:\n", missing_values[missing_values > 0])
```

**Explicación:**  
Este código verifica los valores faltantes en cada columna e imprime solo aquellas que tienen datos faltantes.

---

#### 2.2 Visualizar Distribuciones de cada atributo (no variable objetivo)

Comprender la distribución de cada atributo (no la variable objetivo) puede ayudar a tomar decisiones sobre cómo preprocesar los datos, como si es necesario normalizar las características.

**Código Ejemplo: Graficar las Distribuciones de las Características**

```python
# Graficar la distribución de todas las características en el conjunto de datos
df.hist(bins=30, figsize=(20, 15))
plt.suptitle('Distribuciones de los Atributos')
plt.show()
```

**Explicación:**  
Este código genera histogramas para cada atributo presente en el Dataset, permitiendo visualizar sus distribuciones.

#### 2.3 Explorar las Relaciones Atributo-Objetivo

Para regresión, los diagramas de dispersión pueden ayudar a visualizar las relaciones entre los atributos y la variable objetivo. Para clasificación, los diagramas de caja o violín son útiles para entender las interacciones.

**Código Ejemplo: Relación Atributo-Objetivo (Regresión)**

```python
# Diagrama de dispersión para explorar relaciones atributo-objetivo en datos continuos
for feature in df_california.columns[:-1]:  # Excluir la columna objetivo
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df_california[feature], y=df_california['target'])
    plt.title(f'Relación entre {feature} y la Variable Objetivo')
    plt.show()
```

**Código Ejemplo: Relación Atributo-Objetivo (Clasificación)**

```python
# Diagrama de caja para explorar relaciones atributo-objetivo en clasificación
for feature in df_iris.columns[:-1]:  # Excluir la columna objetivo
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='target', y=feature, data=df_iris, palette='Set2')
    plt.title(f'{feature} vs Variable Objetivo')
    plt.show()
```

**Explicación:**  
Estos diagramas proporcionan una representación visual de cómo cada atributo se relaciona con la variable objetivo.

---

#### 2.4 Verificar Correlaciones

Los mapas de calor de correlación ayudan a identificar multicolinealidad entre los atributos y sus relaciones con la variable objetivo.

**Código Ejemplo: Mapa de Calor de Correlaciones**

```python
# Graficar un mapa de calor de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(df_california.corr(), annot=True, cmap='coolwarm')
plt.title('Mapa de Calor de Correlación entre Atributos')
plt.show()
```

**Explicación:**  
Este mapa de calor destaca cómo se correlacionan los atributos entre sí y con la variable objetivo. Las altas correlaciones entre atributos podrían sugerir multicolinealidad, lo que puede afectar el rendimiento del modelo.