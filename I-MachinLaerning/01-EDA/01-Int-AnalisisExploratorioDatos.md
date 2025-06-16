# Preparación de Datos para el Aprendizaje Automático

## Importancia
La preparación de datos es clave para obtener buenos resultados en el aprendizaje automático. Inicia con la planificación, el tamaño del conjunto de datos y la formulación del problema.  

## Tamaño del conjunto de datos
No hay una cantidad fija de datos necesarios para entrenar un modelo; depende del problema y del algoritmo utilizado. La regla general es recolectar la mayor cantidad posible, aunque "mucho" puede ser un término subjetivo.  

### Ejemplos prácticos  
- **Gmail de Google:** Se usaron 238 millones de mensajes para entrenar sugerencias de respuesta inteligente.  
- **Google Translate:** Requiere billones de ejemplos para su funcionamiento.  
- **Red neuronal para hormigón:** Un profesor logró buenos resultados con solo 630 muestras, demostrando que el tamaño del conjunto de datos depende de la complejidad del problema.  

La clave es ajustar la cantidad de datos a las necesidades del modelo sin sobrecargarlo innecesariamente.

---

# Proceso de Preparación de Datos

Para que un modelo de aprendizaje automático funcione correctamente, los datos deben prepararse mediante las siguientes etapas:

## Transformación de datos
Antes de alimentar un modelo, los datos deben convertirse a un formato adecuado. Esto incluye:  
- **Etiquetado:** Asignar categorías o valores a los datos según el problema a resolver.  
- **Reducción y limpieza:** Eliminar información irrelevante o incorrecta para mejorar la calidad del conjunto de datos.  
- **Muestreo:** Seleccionar un subconjunto representativo de datos cuando el conjunto total es demasiado grande.  

## Limpieza de datos  
Corregir o eliminar valores incorrectos, ausentes o inconsistentes para evitar sesgos en el modelo.  

## Transformación de datos  
Aplicar procesos como normalización, estandarización y codificación para garantizar que los datos estén en un formato uniforme y optimizado para el modelo.

---

# Ingeniería de Características

La **ingeniería de características** es el proceso de transformar datos sin procesar en características relevantes que mejoran el rendimiento de los modelos de aprendizaje automático.

## Principales aspectos
- **Selección de características:** Identificar las variables más relevantes para el modelo y eliminar aquellas que no aportan valor.  
- **Transformación de datos:** Aplicar técnicas como normalización, estandarización y codificación para mejorar la calidad de las características.  
- **Creación de nuevas características:** Generar variables derivadas de los datos existentes para mejorar la capacidad predictiva del modelo.  
- **Reducción de dimensionalidad:** Utilizar métodos como PCA (Análisis de Componentes Principales) para simplificar el conjunto de características sin perder información clave.  

---

# Ejemplos de Ingeniería de Características

### Creación de nuevas características  
- En un conjunto de datos de ventas, en lugar de usar solo el precio y la cantidad, se puede crear una nueva característica: **valor total de la compra** (`precio × cantidad`).  
- En un modelo de predicción de salud, se puede calcular el **Índice de Masa Corporal (IMC)** a partir del peso y la altura.  

### Transformación de características  
- Convertir fechas en valores numéricos, como extraer el **día de la semana** de una fecha de compra para analizar patrones de consumo.  
- Normalizar valores de ingresos para que estén en una escala uniforme y evitar que dominen el modelo.  

### Codificación de variables categóricas  
- Convertir categorías como `"bajo", "medio", "alto"` en valores numéricos (`1, 2, 3`).  
- Aplicar **codificación one-hot** para representar colores de autos como variables binarias (`rojo = [1,0,0]`, `azul = [0,1,0]`, `verde = [0,0,1]`).  