# Testu Sailkatzailea: Spam vs Ham (Logistic Regression)

Este proyecto implementa un sistema de minería de textos en Java utilizando la librería **Weka**. El objetivo es clasificar correos electrónicos en dos categorías: `spam` y `ham` (legítimos), aplicando técnicas de preprocesamiento, vectorización (TF-IDF) y un modelo de Regresión Logística.

El proyecto está dividido en dos módulos principales para cumplir con los roles de **Data Scientist** (entrenamiento y experimentación) y **Erabiltzailea / Cliente** (inferencia en producción).

## 📂 Estructura del Proyecto

* **`SpamClassifierExperimentAdmin.java` (Rol Data Scientist):**
    * Lee los correos crudos desde las carpetas locales.
    * Aplica limpieza de texto mediante expresiones regulares.
    * Vectoriza el texto utilizando `StringToWordVector` (TF-IDF, vocabulario > 500 palabras).
    * Realiza un análisis de sensibilidad (*Fine-Tuning*) probando diferentes valores de penalización `Ridge` usando validación cruzada (10-fold CV) sobre el conjunto de entrenamiento.
    * Entrena el modelo final óptimo empaquetado en un `FilteredClassifier` (asegurando compatibilidad *test-blind*).
    * Genera reportes de calidad (`quality.txt`) y predicciones de test (`predictions.txt`).
    * Exporta el modelo final a disco: `spam_classifier_final.model`.

* **`SpamPredictorUser.java` (Rol Cliente):**
    * Software ligero diseñado para ejecutarse sin interfaz gráfica (vía terminal).
    * Carga el modelo pre-entrenado `spam_classifier_final.model`.
    * Recibe un texto crudo como argumento, aplica la misma vectorización *on-the-fly* y devuelve la predicción instantánea (`spam` o `ham`).

## 🛠️ Requisitos Previos

1.  **Java Development Kit (JDK):** Versión 8 o superior.
2.  **Weka:** El archivo `weka.jar` debe estar disponible en el sistema para compilar y ejecutar el código.
3.  **Datos:** Dos carpetas llamadas `ham` y `spam` en el mismo directorio de ejecución, conteniendo los correos en formato `.txt`.

## 🚀 Compilación

Abre tu terminal, navega a la carpeta del proyecto y compila ambos archivos asegurándote de incluir la ruta a tu archivo `weka.jar` en el classpath (modifica `ruta/a/weka.jar` según tu sistema).

**En Linux/Mac:**
```bash
javac -cp ".:ruta/a/weka.jar" SpamClassifierExperiment.java SpamPredictor.java
