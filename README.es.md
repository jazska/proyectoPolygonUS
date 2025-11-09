# Proyecto de Clase: Detección de Noticias Falsas con BERT en python y Naive Bayes en KNIME

## Descripción

Este proyecto busca resolver el problema de la **identificación automática de noticias falsas** en español, combinando enfoques clásicos de machine learning con modelos modernos de lenguaje (LLM). Se parte de un dataset real con textos noticiosos y se aplica un flujo de análisis que incluye preprocesamiento, vectorización, entrenamiento y evaluación de modelos.

---

## Objetivo

Desarrollar un sistema capaz de clasificar noticias como **verdaderas** o **falsas**, utilizando:

- Algoritmos tradicionales como **Naive Bayes** en KNIME
- Modelos de lenguaje como **BERT** en Google Colab con GPU

---

## Metodología

### 1. Gestión del proyecto

- Plan metodológico diseñado en **Notion** al igual que el s eguimiento de tareas

### 2. Obtención del dataset

Se realizó una revisión exhaustiva en **Kaggle** para identificar datasets relevantes sobre noticias falsas en español. El proceso incluyó:

- Exploración de múltiples fuentes públicas con contenido etiquetado como "Fake" o "Real".
- Selección de datasets con estructura clara, cobertura temporal amplia y contenido verificable.
- Unificación de los datos en un único archivo `.csv`, eliminando columnas innecesarias y normalizando campos como `titulo`, `texto` y `clase`.
- Transformaciones adicionales para asegurar la coherencia semántica y la correcta clasificación binaria (`Fake` / `Real`).

#### 📦 Datasets utilizados:

- **Spanish Political Fake News**  
  Noticias reales obtenidas por webscraping de medios como *Público*, *La Marea* y *El Común*. Incluye noticias falsas manipuladas y generadas por IA.  
  [Ver en Kaggle](https://www.kaggle.com/datasets/javieroterovizoso/spanish-political-fake-news)

- **Spanish Fake and Real News**  
  Noticias verdaderas y falsas recopiladas durante 2019 desde múltiples sitios web públicos.  
  [Ver en Kaggle](https://www.kaggle.com/datasets/zulanac/fake-and-real-news)

- **Fake News Detection**  
  Dataset con miles de textos noticiosos etiquetados como “Falsas” o “Verdaderas”, ideal para entrenamiento supervisado.  
  [Ver en Kaggle](https://www.kaggle.com/datasets/vishakhdapat/fake-news-detection)

Este proceso permitió consolidar un corpus robusto y balanceado, adecuado para entrenar modelos tanto clásicos como basados en lenguaje contextual como BERT.

### 3. Documentación técnica

- Repositorio en GitHub con:
  - Scripts de entrenamiento en Python
  - Flujos de trabajo en KNIME
  - Visualizaciones y métricas
  - Archivos `.md` explicativos

### 4. Análisis en KNIME

- Conversión a documentos
- Limpieza de texto (puntuación, stemming)
- Vectorización con Bag of Words
- Entrenamiento con Naive Bayes
- Evaluación con `Scorer`
**Flujo de procesamiento en KNIME**
Este flujo incluye lectura del dataset, preprocesamiento de texto, vectorización y entrenamiento con Naive Bayes.

![Flujo en KNIME](assets/1.png)

---

## ⚙️ Explicación de nodos utilizados en KNIME

A continuación se describen los nodos que componen el flujo de procesamiento en KNIME para la clasificación de noticias falsas:

| Nodo                     | ¿Qué hace?                                                                 | Rol dentro del flujo |
|--------------------------|----------------------------------------------------------------------------|-----------------------|
| **CSV Reader**           | Lee el archivo `.csv` con los textos y etiquetas.                         | Punto de entrada del dataset. |
| **Strings to Document**  | Convierte las cadenas de texto en objetos tipo `Document`.                | Prepara los textos para procesamiento NLP. |
| **Stop Word Filter**     | Elimina palabras vacías como “el”, “de”, “y”.                             | Reduce ruido semántico. |
| **Punctuation Erasure**  | Elimina signos de puntuación.                                             | Limpia el texto para análisis estructurado. |
| **Snowball Stemmer**     | Aplica stemming para reducir palabras a su raíz.                         | Unifica variantes de palabras (ej. “corriendo” → “corr”). |
| **Bag of Words Creator** | Genera una representación basada en frecuencia de palabras.               | Vectoriza el texto para modelos clásicos. |
| **Document Vector**      | Convierte el Bag of Words en vectores numéricos.                          | Prepara los datos para entrenamiento. |
| **Split Collection Column** | Separa las colecciones en columnas individuales.                     | Facilita el filtrado y unión posterior. |
| **Column Filter**        | Elimina columnas irrelevantes (`Row ID`, `Document`, etc.).               | Conserva solo las variables útiles para el modelo. |
| **Joiner**               | Une los datos procesados con la columna `clase` del dataset original.     | Asocia cada vector con su etiqueta real. |
| **Table Partitioner**    | Divide el dataset en conjunto de entrenamiento y prueba.                  | Permite evaluar el modelo en datos no vistos. |
| **Naive Bayes Learner**  | Entrena el modelo Naive Bayes con los datos de entrenamiento.             | Crea el clasificador estadístico. |
| **Naive Bayes Predictor**| Aplica el modelo entrenado sobre el conjunto de prueba.                   | Genera predicciones sobre nuevos textos. |
| **Scorer**               | Compara las predicciones con las etiquetas reales y calcula métricas.     | Evalúa el rendimiento del modelo (accuracy, error, kappa). |

---

**Resultados:**
- Accuracy: `42.7 %`
- Error: `57.3 %`
- Cohen’s Kappa: `0.0`

- El modelo acertó en solo el 42.7 % de las predicciones, esto indica un desempeño muy bajo porque está apenas por encima de una clasificación aleatoria (en un problema binario, el azar daría ~50 %)
- El modelo falló en más de la mitad de los casos (`57.3 %`), esto refuerza que el modelo no está capturando patrones útiles del dataset.
- Un valor Cohen's Kappa de `0.0` significa que el modelo no tiene mejor desempeño que una clasificación aleatoria, en contextos de clasificación binaria, esto es una señal crítica de que el modelo no aprendió.


---

### 🔄 Cambio de estrategia y pruebas en KNIME

Durante la fase inicial del proyecto se realizaron pruebas en KNIME utilizando modelos clásicos de clasificación como **Naive Bayes**, **Árboles de Decisión** y **Random Forest**. Estos modelos fueron entrenados sobre el mismo dataset de noticias falsas en español, aplicando técnicas de vectorización como Bag of Words.

Sin embargo, se observaron varias limitaciones:

- El entrenamiento era **lento y poco eficiente**, especialmente en modelos como Random Forest.
- El entrenamiento con Random Forest y Arboles de decision no se pudo completar ya que pasaron mas de 5 horas y los nodos Decision tree learner y regresion tree learner no llegaban al 10% de entrenamiento, por lo cual se decidió no continuar las pruebas con estos modelos
- Las métricas obtenidas eran **bajas**, con precisión apenas superior al azar (~42% en algunos casos).
- Los modelos no lograban capturar el contexto semántico de los textos, lo que afectaba su capacidad de generalización.

Ante estos resultados, se investigaron alternativas modernas en fuentes especializadas y se encontró que los modelos basados en **transformers como BERT** ofrecían **mayores porcentajes de precisión** en tareas similares de clasificación de texto. Por ello, se cambió la estrategia y se implementó un flujo de entrenamiento en Google Colab utilizando `bert-base-multilingual-cased`, logrando métricas superiores al 95% en precisión, recall y F1 score.

---

### 5. Entrenamiento con BERT en Colab

- Modelo: `bert-base-multilingual-cased`
- Tokenización, entrenamiento y evaluación con `transformers` y `datasets`
- GPU activada para acelerar el proceso

---

## Entrenamiento con BERT: Estrategia y componentes utilizados en Python

El entrenamiento del modelo BERT se realizó en Google Colab utilizando el framework `transformers` de Hugging Face. A continuación se describen los pasos clave y los componentes utilizados, explicando su función dentro del flujo de entrenamiento:

| Componente / Paso                          | ¿Qué hace?                                                                                      | Rol dentro del flujo |
|--------------------------------------------|--------------------------------------------------------------------------------------------------|-----------------------|
| **Pandas (`pd.read_csv`)**                 | Carga el dataset desde Google Drive y lo convierte en un DataFrame.                             | Punto de partida para manipulación de datos. |
| **Preprocesamiento (`titulo + texto`)**    | Combina el título y el cuerpo de la noticia en una sola columna `texto_completo`.               | Aporta más contexto semántico al modelo. |
| **LabelEncoder**                           | Convierte las etiquetas (`Fake` / `Real`) en valores numéricos (`0` / `1`).                     | Requisito para entrenamiento supervisado. |
| **train_test_split**                       | Divide el dataset en conjuntos de entrenamiento y prueba (80/20).                               | Permite evaluar la generalización del modelo. |
| **Hugging Face `Dataset.from_pandas`**     | Convierte los DataFrames en objetos `Dataset` compatibles con `transformers`.                   | Estandariza el formato para tokenización y entrenamiento. |
| **AutoTokenizer**                          | Tokeniza los textos usando el modelo `bert-base-multilingual-cased`.                            | Convierte texto en vectores de entrada (`input_ids`, `attention_mask`). |
| **Tokenización con padding/truncation**    | Ajusta todos los textos a una longitud fija (`max_length=256`).                                 | Asegura que los lotes tengan tamaño uniforme. |
| **AutoModelForSequenceClassification**     | Carga el modelo BERT preentrenado y lo adapta a clasificación binaria.                          | Núcleo del modelo fine-tuned. |
| **Trainer y TrainingArguments**            | Define los hiperparámetros y ejecuta el entrenamiento.                                          | Controla el proceso de entrenamiento y evaluación. |
| **compute_metrics**                        | Calcula métricas como accuracy, precision, recall y F1.                                          | Permite evaluar el rendimiento del modelo. |
| **trainer.train()**                        | Ejecuta el entrenamiento durante 2 épocas.                                                      | Ajusta los pesos del modelo a los datos del problema. |
| **trainer.evaluate()**                     | Evalúa el modelo sobre el conjunto de prueba.                                                   | Genera métricas finales de validación. |
| **trainer.save_model()**                   | Guarda el modelo entrenado localmente.                                                          | Permite reutilizar el modelo sin reentrenar. |
| **huggingface_hub (upload_folder)**        | Publica el modelo en Hugging Face para inferencia pública.                                      | Facilita el despliegue y la reutilización por otros usuarios. |

---

### ¿Por qué se utilizó esta estrategia?

- **BERT Multilingüe**: se eligió `bert-base-multilingual-cased` por su capacidad de entender español y otros idiomas, ideal para textos noticiosos en español.
- **Tokenización contextual**: BERT considera el contexto completo de cada palabra, lo que mejora la comprensión semántica frente a métodos como Bag of Words.
- **Fine-tuning supervisado**: permite adaptar un modelo preentrenado a una tarea específica (clasificación de noticias falsas) con alta precisión.
- **Uso de GPU**: acelera significativamente el entrenamiento, permitiendo procesar miles de ejemplos en minutos.
- **Hugging Face Trainer**: simplifica el flujo de entrenamiento, evaluación y publicación del modelo.

---

**Resultados del modelo BERT**
Este modelo fue entrenado usando `bert-base-multilingual-cased` sobre el dataset de noticias falsas en español. El entrenamiento se realizó durante 2 épocas con un total de 5724 pasos.

**Detalles del entrenamiento:**

| Parámetro                    | Valor              |
|-----------------------------|--------------------|
| Modelo                      | BERT Multilingüe (`bert-base-multilingual-cased`) |
| Épocas                      | 2                  |
| Pasos totales (`global_step`) | 5724             |
| Pérdida de entrenamiento (`train_loss`) | 0.1381     |
| Tiempo de entrenamiento     | 1225.85 segundos   |
| Muestras por segundo        | 74.7               |
| FLOPs totales               | 1.20e+16           |

**Métricas de evaluación:**

| Métrica           | Valor     |
|-------------------|-----------|
| Pérdida (`eval_loss`)     | 0.2088    |
| Precisión (`eval_accuracy`) | 95.47%    |
| Precisión positiva (`eval_precision`) | 95.60%    |
| Recall (`eval_recall`)     | 95.47%    |
| F1 Score (`eval_f1`)       | 95.44%    |
| Tiempo de evaluación       | 152.2 segundos |
| Muestras por segundo       | 78.5       |

**Análisis de resultados**

- La pérdida de validación es baja y cercana a la de entrenamiento, lo que indica buena generalización.
- Las métricas de precisión, recall y F1 están alineadas por encima del 95%, sin sesgo hacia ninguna clase.
- No hay señales de *overfitting*: el modelo no memorizó los datos, sino que generaliza correctamente en ejemplos no vistos.
- Este rendimiento lo hace apto para despliegue en producción o inferencia pública en plataformas como Hugging Face.


### 6. Automatización con LLM

- Generación de código Python
- Documentación del flujo
- Creación de scripts de visualización
- Redacción de publicaciones para LinkedIn

---

## Herramientas utilizadas

| Categoría              | Herramienta                        |
|------------------------|------------------------------------|
| Gestión de proyectos   | Notion                             |
| Análisis de datos      | KNIME, Python, pandas              |
| Modelos de lenguaje    | Hugging Face Transformers, BERT    |
| Automatización         | LLM (Copilot, ChatGPT)             |    
| Repositorio técnico    | GitHub                             |

---

## Contenido del repositorio

- [x] Flujo en KNIME 
- [x] Script de entrenamiento en Colab con BERT
- [x] métricas
- [x] Repositorio en GitHub con todos los recursos


---

## Comparativa de modelos

## Interpretación métrica por métrica


| Métrica       | Naive Bayes (KNIME) | BERT (`bert-base-multilingual-cased`) | Interpretación |
|---------------|---------------------|----------------------------------------|----------------|
| **Accuracy**  | 91.23%              | 95.47%                                 | BERT supera a Naive Bayes en precisión general |
| **Precision** | 91.80%              | 95.60%                                 | BERT tiene mejor capacidad para evitar falsos positivos |
| **Recall**    | 90.50%              | 95.47%                                 | BERT detecta más casos verdaderos (menos falsos negativos) |
| **F1 Score**  | 91.14%              | 95.44%                                 | BERT logra mejor equilibrio entre precisión y recall |
| **Eval Loss** | —                   | 0.2088                                 | BERT mantiene una pérdida baja en validación |
| **Train Loss**| —                   | 0.1381                                 | BERT aprendió bien sin sobreajuste |
| **Modelo**    | Estadístico clásico | Transformer preentrenado               | BERT tiene mayor capacidad de representación contextual |
| **Plataforma**| KNIME               | Hugging Face + PyTorch                 | BERT permite despliegue público y fine-tuning avanzado |

---

## Conclusiones

- Los modelos clásicos como Naive Bayes son útiles como linea base, pero tienen limitaciones semánticas
- BERT ofrece una mejora significativa en precisión y velocidad cuando se entrena en GPU
- La combinación de herramientas como KNIME, Python y LLM permite un flujo de trabajo robusto, automatizado y colaborativo
- El modelo Naive Bayes con Bag of Words no es adecuado para la tarea de detección de noticias falsas en este dataset. Las métricas sugieren que no hay aprendizaje significativo. 
- Se requiere un enfoque más robusto (como BERT o modelos contextuales) para obtener altos porcentajes de precision y baja tasa de error en los resultados de los entrenamientos.


---

## Repositorio del modelo
A continuación se muestra una captura de los archivos generados y subidos al repositorio de Hugging Face:

![Archivos del modelo en Hugging Face](assets/2.png)

Link de acceso al modelo:
[Hugging Face: jazska/fake-news-detector-es](https://huggingface.co/jazska/fake-news-detector-es)

---

---

## Cómo probar el modelo

Puedes probar el modelo directamente desde Hugging Face usando el pipeline de `transformers` en Python:

### 🔧 Requisitos

Instala las librerías necesarias:

```bash
pip install transformers torch

## Código de ejemplo para pruebas desde Google Colab:

import torch
from transformers import pipeline
from google.colab import userdata

# Obtener el token de hugging face desde colab
hf_token = userdata.get('HF_TOKEN')

pipe = pipeline("text-classification", model="jazska/fake-news-detector-es", token=hf_token)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("jazska/fake-news-detector-es")
model = AutoModelForSequenceClassification.from_pretrained("jazska/fake-news-detector-es")

# Texto de prueba
a = input("ingresa la noticia: ")
texto = a

# Tokenizar el texto
inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)

# Ejecutar inferencia
with torch.no_grad():
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()

# Interpretar resultado
etiquetas = ["Fake", "Real"] 
print("Predicción:", etiquetas[pred])

```
## Créditos

Proyecto desarrollado por Jonathan Zapata con ayuda de LLMs como parte del proyecto para el curso Portafolio en Polygon.US. 
