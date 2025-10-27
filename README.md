# Proyecto de Clase: Detección de Noticias Falsas con BERT y KNIME

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

- Dataset consolidado en CSV con columnas: `titulo`, `texto`, `clase`
- Fuentes: Kaggle, Internet y datos propios

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

**Resultados:**
- Accuracy: `42.7 %`
- Error: `57.3 %`
- Cohen’s Kappa: `0.0`

- El modelo acertó en solo el 42.7 % de las predicciones, esto indica un desempeño muy bajo porque está apenas por encima de una clasificación aleatoria (en un problema binario, el azar daría ~50 %)
- El modelo falló en más de la mitad de los casos (`57.3 %`), esto refuerza que el modelo no está capturando patrones útiles del dataset.
- Un valor Cohen's Kappa de `0.0` significa que el modelo no tiene mejor desempeño que una clasificación aleatoria, en contextos de clasificación binaria, esto es una señal crítica de que el modelo no aprendió.

### 5. Entrenamiento con BERT en Colab

- Modelo: `bert-base-multilingual-cased`
- Tokenización, entrenamiento y evaluación con `transformers` y `datasets`
- GPU activada para acelerar el proceso

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

## Créditos

Proyecto desarrollado como parte del curso de análisis de datos y automatización con LLM.  
Incluye integración de herramientas de gestión, visualización, entrenamiento y publicación técnica.
