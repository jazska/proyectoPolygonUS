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

**Resultados:**
- Accuracy: `42.7 %`
- Error: `57.3 %`
- Cohen’s Kappa: `0.0`

### 5. Entrenamiento con BERT en Colab

- Modelo: `bert-base-multilingual-cased`
- Tokenización, entrenamiento y evaluación con `transformers` y `datasets`
- GPU activada para acelerar el proceso

**Resultados:**
- Accuracy: `97.2 %`
- F1 Score: `0.972`
- Precision y Recall: `0.972`
- Tiempo de entrenamiento: `36 segundos`

### 6. Automatización con LLM

- Generación de código Python
- Documentación del flujo
- Creación de scripts de visualización
- Redacción de publicaciones para LinkedIn

---

## Herramientas utilizadas

| Categoría              | Herramienta                        |
|------------------------|------------------------------------|
| Gestión de proyectos   | Notion, Trello, ClickUP            |
| Análisis de datos      | KNIME, Python, pandas              |
| Modelos de lenguaje    | Hugging Face Transformers, BERT    |
| Automatización         | LLM (Copilot, ChatGPT, etc.)       |
| Visualización          | Matplotlib, plot_tree, KNIME views |
| Repositorio técnico    | GitHub                             |

---

## Entregables

- [x] Flujo en KNIME documentado
- [x] Script de entrenamiento en Colab con BERT
- [x] Visualizaciones de métricas
- [x] Publicación en LinkedIn
- [x] Repositorio en GitHub con todos los recursos
- [x] Pitch en video explicando la propuesta

---

## Comparativa de modelos

| Métrica              | Naive Bayes (KNIME) | BERT (Colab) |
|----------------------|---------------------|--------------|
| Accuracy             | 42.7 %              | 97.2 %       |
| Error                | 57.3 %              | 11.2 %       |
| F1 Score             | —                   | 0.972        |
| Cohen’s Kappa        | 0.0                 | —            |

---

## Conclusiones

- Los modelos clásicos como Naive Bayes son útiles como baseline, pero tienen limitaciones semánticas
- BERT ofrece una mejora significativa en precisión y velocidad cuando se entrena en GPU
- La combinación de herramientas como KNIME, Python y LLM permite un flujo de trabajo robusto, automatizado y colaborativo
- Documentar y publicar el proyecto en GitHub y LinkedIn fortalece la visibilidad profesional y el aprendizaje técnico

---

## Repositorio del modelo

📁 [Hugging Face: jazska/fake-news-detector-es](https://huggingface.co/jazska/fake-news-detector-es)

---

## Créditos

Proyecto desarrollado como parte del curso de análisis de datos y automatización con LLM.  
Incluye integración de herramientas de gestión, visualización, entrenamiento y publicación técnica.
