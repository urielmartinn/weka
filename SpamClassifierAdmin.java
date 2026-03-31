import weka.core.*;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stopwords.Rainbow;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.tokenizers.NGramTokenizer;
import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Clase principal para el rol de Data Scientist (Administrador).
 * Implementa un pipeline completo de Machine Learning para clasificar correos en Spam o Ham.
 * Se encarga de:
 * 1. Carga de datos.
 * 2. División Hold-out estratificada (Train/Test).
 * 3. Preprocesamiento NLP (Stopwords, Stemming) y Vectorización (BoW, TF-IDF, N-Gramas).
 * 4. Parameter Fine-Tuning (Grid Search mediante 10-fold Cross-Validation).
 * 5. Evaluación final y serialización del modelo óptimo usando FilteredClassifier para evitar el Test-Blind.
 */
public class SpamClassifierAdmin {

    // ========== PARÁMETROS FIJOS (Panel de control de experimentos) ==========
    
    /** Array de valores del parámetro de regularización Ridge (L2) a evaluar en el Grid Search. */
    private static final double[] RIDGE_VALUES = {1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0};
    
    /** Límite de palabras del diccionario. Actúa como método empírico de selección de atributos. */
    private static final int WORDS_TO_KEEP = 1000;
    
    /** Proporción del conjunto de datos destinado a entrenamiento (ej. 0.8 = 80%). */
    private static final double TRAIN_RATIO = 0.8;
    
    /** Semilla para garantizar la reproducibilidad de la aleatoriedad en las particiones. */
    private static final long RANDOM_SEED = 12345L;

    // --- Aurreprozesamendua (Preprocesamiento NLP) ---
    /** Si es true, aplica el diccionario Rainbow para eliminar palabras vacías. */
    private static final boolean USE_STOPWORDS = true;
    
    /** Si es true, aplica LovinsStemmer para reducir las palabras a su raíz lematizada. */
    private static final boolean USE_STEMMER = true;

    // --- Bektorizazioa (Representación Vectorial) ---
    /** Si es true, cuenta las frecuencias (TF). Si es false, usa representación BoW binaria (0/1). */
    private static final boolean OUTPUT_WORD_COUNTS = true;
    
    /** Si es true, aplica una transformación logarítmica a las frecuencias de las palabras. */
    private static final boolean USE_TF = true;
    
    /** Si es true, aplica la penalización Inverse Document Frequency para palabras muy comunes. */
    private static final boolean USE_IDF = true;
    
    /** Si es true, el tokenizador extraerá tanto unigramas como bigramas secuenciales. */
    private static final boolean USE_BIGRAMS = false;
    
    /**
     * Clase auxiliar interna para modelar y almacenar los metadatos de un correo electrónico.
     */
    private static class Email {
        String text;
        String label;
        String author;
        Date date;

        /**
         * Construye un nuevo objeto Email.
         *
         * @param text   El contenido del correo en formato crudo.
         * @param label  La clase real del correo ("ham" o "spam").
         * @param author El autor extraído del nombre del archivo.
         * @param date   La fecha extraída del nombre del archivo.
         */
        Email(String text, String label, String author, Date date) {
            this.text = text;
            this.label = label;
            this.author = author;
            this.date = date;
        }
    }

    /**
     * Método principal que ejecuta el pipeline secuencial del experimento.
     *
     * @param args Argumentos de la terminal. args[0] = carpeta ham, args[1] = carpeta spam.
     * @throws Exception Si ocurre un error en la lectura de archivos, filtrado o modelado en Weka.
     */
    public static void main(String[] args) throws Exception {
        // Leer carpetas desde argumentos o usar valores por defecto
        String hamFolder = (args.length > 0) ? args[0] : "ham";
        String spamFolder = (args.length > 1) ? args[1] : "spam";
        String outputQuality = "quality.txt";
        String outputPredictions = "predictions.txt";

        // 1. Cargar todos los correos
        System.out.println("Cargando correos desde " + hamFolder + " y " + spamFolder + "...");
        List<Email> allEmails = new ArrayList<>();
        loadEmails(hamFolder, "ham", allEmails);
        loadEmails(spamFolder, "spam", allEmails);
        System.out.println("Total correos cargados: " + allEmails.size());

        // 2. Dividir en entrenamiento y test
        System.out.println("Dividiendo en train/test (ratio=" + TRAIN_RATIO + ")...");
        List<Email> trainEmails = new ArrayList<>();
        List<Email> testEmails = new ArrayList<>();
        splitData(allEmails, trainEmails, testEmails);
        System.out.println("Entrenamiento: " + trainEmails.size() + " correos");
        System.out.println("Test: " + testEmails.size() + " correos");

        // 3. Crear datasets Weka (texto crudo)
        Instances trainRaw = createInstances(trainEmails, true);
        Instances testRaw = createInstances(testEmails, true); // con etiquetas para evaluación
        trainRaw.setClassIndex(trainRaw.numAttributes() - 1);
        testRaw.setClassIndex(testRaw.numAttributes() - 1);

        // 4. Configurar filtro de vectorización
        System.out.println("Configurando vectorización (words=" + WORDS_TO_KEEP + ", TF=" + USE_TF + ", IDF=" + USE_IDF + ")...");
        StringToWordVector filter = new StringToWordVector();
        filter.setLowerCaseTokens(true);
        filter.setWordsToKeep(WORDS_TO_KEEP);

        // Configuramos la representación espacial (Binario, TF, o TF-IDF)
        filter.setOutputWordCounts(OUTPUT_WORD_COUNTS);
        filter.setTFTransform(USE_TF);
        filter.setIDFTransform(USE_IDF);

        // APLICANDO PREPROCESAMIENTO NLP ---
        if (USE_STOPWORDS) {
            System.out.println(" -> Aplicando filtro de Stopwords (Rainbow)...");
            filter.setStopwordsHandler(new Rainbow());
        }
        if (USE_STEMMER) {
            System.out.println(" -> Aplicando Stemmer (LovinsStemmer)...");
            filter.setStemmer(new LovinsStemmer());
        }

        if (USE_BIGRAMS) {
            System.out.println(" -> Aplicando Tokenizador de Bigramas...");
            NGramTokenizer tokenizer = new NGramTokenizer();
            tokenizer.setNGramMinSize(1); // Unigramas
            tokenizer.setNGramMaxSize(2); // Bigramas
            filter.setTokenizer(tokenizer);
        }

        // TABLA DIMENSIONALIDAD (Corregida)
        System.out.println("\n=== TAULA: Datuen dimentsionalitatearen eboluzioa ===");
        System.out.println(String.format("%-35s | %-12s | %-12s", "Prozesua / Egoera", "Instantziak", "Atributuak"));
        System.out.println(String.format("%-35s | %-12d | %-12d", 
            "1. Datu gordinak (Raw Text)", trainRaw.numInstances(), trainRaw.numAttributes()));
        
        filter.setInputFormat(trainRaw); // Aplicar temporalmente solo para medir
        Instances trainTabular = Filter.useFilter(trainRaw, filter);

        System.out.println(String.format("%-35s | %-12d | %-12d", 
            "2. Bektorizazioa (BoW/TF-IDF)", trainTabular.numInstances(), trainTabular.numAttributes()));
        System.out.println("====================================================================\n");

        // 5. Entrenar regresión logística con FILTERED CLASSIFIER (¡CRÍTICO!)
        System.out.println("RIDGE");
        System.out.println("10-fold Cross-Validation en Train");
        
        double bestRidge = RIDGE_VALUES[0];
        double bestFMeasure = -1.0;

        // Bucle iterativo para probar cada valor
        for (double currentRidge : RIDGE_VALUES) {
            Logistic tempLogistic = new Logistic();
            tempLogistic.setOptions(new String[]{"-R", String.valueOf(currentRidge), "-M", "500"});

            FilteredClassifier tempFc = new FilteredClassifier();
            tempFc.setFilter(filter); // Usamos el filtro configurado en el Paso 4
            tempFc.setClassifier(tempLogistic);

            // Evaluamos usando Cross-Validation sobre el TRAIN
            Evaluation cvEval = new Evaluation(trainRaw);
            cvEval.crossValidateModel(tempFc, trainRaw, 10, new Random(RANDOM_SEED));

            // Extraemos la métrica (F-Measure de la clase Spam, que suele ser el índice 1)
            // Buscamos dinámicamente el índice de la clase "spam"
            int spamIndex = trainRaw.classAttribute().indexOfValue("spam");
            double fMeasureSpam = cvEval.fMeasure(spamIndex);
            double accuracy = cvEval.pctCorrect();

            System.out.println(String.format("Ridge: %1.0e | Accuracy: %5.2f%% | F-Measure (Spam): %.4f", 
                                              currentRidge, accuracy, fMeasureSpam));

            // Guardamos el mejor
            if (fMeasureSpam > bestFMeasure) {
                bestFMeasure = fMeasureSpam;
                bestRidge = currentRidge;
            }
        }

        System.out.println("MEJOR RIDGE ENCONTRADO: " + bestRidge + " (F-Measure: " + bestFMeasure + ")");

        // 6. Entrenar el modelo FINAL con el MEJOR Ridge usando TODOS los datos de Train
        System.out.println("Entrenando regresión logística FINAL empaquetada (ridge=" + bestRidge + ")...");
        Logistic finalLogistic = new Logistic();
        finalLogistic.setOptions(new String[]{"-R", String.valueOf(bestRidge), "-M", "500"});

        FilteredClassifier finalFc = new FilteredClassifier();
        finalFc.setFilter(filter);
        finalFc.setClassifier(finalLogistic);
        
        finalFc.buildClassifier(trainRaw);

        // 7. Evaluar en test
        System.out.println("Evaluando en test...");
        Evaluation eval = new Evaluation(trainRaw);
        // ¡OJO! Evaluamos pasándole el test EN CRUDO
        eval.evaluateModel(finalFc, testRaw); 
        saveQualityReport(outputQuality, eval);

        // 8. Generar predicciones sobre test y GUARDAR MODELO PARA EL CLIENTE
        System.out.println("Generando predicciones y guardando modelo...");
        generatePredictions(testRaw, finalFc, outputPredictions);
        weka.core.SerializationHelper.write("spam_classifier_final.model", finalFc);

        System.out.println("Proceso completado. Archivos generados:");
        System.out.println(" - " + outputQuality);
        System.out.println(" - " + outputPredictions);
        System.out.println(" - spam_classifier_final.model (¡Listo para el cliente!)");
    }

    /**
     * Carga todos los correos electrónicos de un directorio especificado y extrae sus metadatos.
     *
     * @param folder Ruta de la carpeta que contiene los archivos .txt.
     * @param label  Etiqueta a asignar a los correos de esta carpeta ("ham" o "spam").
     * @param list   Lista de objetos Email donde se irán añadiendo los correos leídos.
     * @throws Exception Si hay problemas de E/S al leer el directorio.
     */
    private static void loadEmails(String folder, String label, List<Email> list) throws Exception {
        File dir = new File(folder);
        if (!dir.exists() || !dir.isDirectory()) {
            System.err.println("La carpeta no existe: " + folder);
            return;
        }
        File[] files = dir.listFiles((d, name) -> name.endsWith(".txt"));
        if (files == null) return;

        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
        for (File f : files) {
            String content = readFile(f);
            String name = f.getName();
            String[] parts = name.split("\\.");
            String author = (parts.length >= 3) ? parts[2] : "desconocido";
            Date date = null;
            if (parts.length >= 2) {
                try { date = sdf.parse(parts[1]); } catch (Exception e) {}
            }
            content = cleanText(content);
            list.add(new Email(content, label, author, date));
        }
    }

    /**
     * Lee el contenido completo de un archivo de texto plano.
     *
     * @param f Archivo a leer.
     * @return El contenido del archivo como una cadena de texto (String).
     * @throws IOException Si ocurre un error de lectura.
     */
    private static String readFile(File f) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line;
            while ((line = br.readLine()) != null) {
                sb.append(line).append("\n");
            }
        }
        return sb.toString();
    }

    /**
     * Realiza una limpieza básica del texto usando expresiones regulares.
     * Elimina URLs, direcciones de email, números y caracteres especiales no alfabéticos.
     *
     * @param text El texto crudo original.
     * @return El texto estandarizado en minúsculas y sin ruido estructurado.
     */
    private static String cleanText(String text) {
        // Eliminar emails, URLs, números, signos de puntuación
        text = text.replaceAll("\\S+@\\S+\\.\\S+", " ");
        text = text.replaceAll("(http|https)://\\S+", " ");
        text = text.replaceAll("\\b\\d+\\b", " ");
        text = text.replaceAll("[^a-zA-Záéíóúñü' ]", " ");
        text = text.toLowerCase();
        text = text.replaceAll("\\s+", " ").trim();
        return text;
    }

    /**
     * Divide la lista completa de correos en conjuntos de entrenamiento y test
     * manteniendo las proporciones (Hold-out estratificado manualmente) y aleatorizando el orden.
     *
     * @param all   Lista completa de todos los correos.
     * @param train Lista vacía donde se insertarán los correos de entrenamiento.
     * @param test  Lista vacía donde se insertarán los correos de test.
     */
    private static void splitData(List<Email> all, List<Email> train, List<Email> test) {
        List<Email> hamList = new ArrayList<>();
        List<Email> spamList = new ArrayList<>();
        for (Email e : all) {
            if (e.label.equals("ham")) hamList.add(e);
            else spamList.add(e);
        }
        Collections.shuffle(hamList, new Random(RANDOM_SEED));
        Collections.shuffle(spamList, new Random(RANDOM_SEED));
        int hamTrainSize = (int) Math.round(hamList.size() * TRAIN_RATIO);
        int spamTrainSize = (int) Math.round(spamList.size() * TRAIN_RATIO);
        train.addAll(hamList.subList(0, hamTrainSize));
        train.addAll(spamList.subList(0, spamTrainSize));
        test.addAll(hamList.subList(hamTrainSize, hamList.size()));
        test.addAll(spamList.subList(spamTrainSize, spamList.size()));
        Collections.shuffle(train, new Random(RANDOM_SEED));
        Collections.shuffle(test, new Random(RANDOM_SEED));
    }

    /**
     * Convierte una lista de objetos Email de Java a un objeto Instances (Dataset) de Weka.
     * Crea una estructura de datos de solo 2 atributos: "text" (String) y "class" (Nominal).
     *
     * @param emails    Lista de correos a transformar.
     * @param withClass Si es true, asigna la etiqueta real; si es false, la deja como '?' (desconocida).
     * @return El conjunto de datos Instances preparado para entrar al FilteredClassifier.
     */
    private static Instances createInstances(List<Email> emails, boolean withClass) {
        ArrayList<Attribute> atts = new ArrayList<>();
        atts.add(new Attribute("text", (ArrayList<String>) null));
        if (withClass) {
            ArrayList<String> classValues = new ArrayList<>();
            classValues.add("ham");
            classValues.add("spam");
            atts.add(new Attribute("class", classValues));
        } else {
            ArrayList<String> dummy = new ArrayList<>();
            dummy.add("?");
            atts.add(new Attribute("class", dummy));
        }
        Instances data = new Instances("emails", atts, emails.size());
        if (withClass) data.setClassIndex(data.numAttributes() - 1);
        else data.setClassIndex(data.numAttributes() - 1);

        for (Email e : emails) {
            double[] vals = new double[data.numAttributes()];
            vals[0] = data.attribute(0).addStringValue(e.text);
            if (withClass) {
                vals[1] = data.attribute(1).indexOfValue(e.label);
            } else {
                vals[1] = 0;
            }
            data.add(new DenseInstance(1.0, vals));
        }
        return data;
    }

    /**
     * Extrae las métricas de calidad de la Evaluación de Weka y las guarda en un reporte de texto.
     *
     * @param filename Nombre del archivo de salida (ej. "quality.txt").
     * @param eval     El objeto Evaluation de Weka que contiene los resultados del Test.
     * @throws IOException Si falla la escritura en disco.
     */
    private static void saveQualityReport(String filename, Evaluation eval) throws IOException {
        try (BufferedWriter w = new BufferedWriter(new FileWriter(filename))) {
            w.write("=== ESTIMACIÓN DE CALIDAD ===\n");
            w.write("Evaluación sobre conjunto de test (no usado en entrenamiento)\n");
            w.write("Precisión (accuracy): " + eval.pctCorrect() + "%\n");
            w.write("Kappa: " + eval.kappa() + "\n");
            w.write("F1 ponderado: " + eval.weightedFMeasure() + "\n");
            w.write("\nMatriz de confusión:\n");
            w.write(eval.toMatrixString());
            w.write("\n\nMétricas por clase:\n");
            w.write("Clase ham:\n");
            w.write("  Precisión: " + eval.precision(0) + "\n");
            w.write("  Recall: " + eval.recall(0) + "\n");
            w.write("  F1: " + eval.fMeasure(0) + "\n");
            w.write("Clase spam:\n");
            w.write("  Precisión: " + eval.precision(1) + "\n");
            w.write("  Recall: " + eval.recall(1) + "\n");
            w.write("  F1: " + eval.fMeasure(1) + "\n");
        }
    }

    /**
     * Ejecuta el modelo sobre un conjunto de test y guarda las etiquetas predichas línea por línea.
     *
     * @param testData Dataset de test (en crudo).
     * @param model    El modelo clasificador ya entrenado (FilteredClassifier).
     * @param outFile  Archivo de salida para el listado de predicciones.
     * @throws Exception Si el modelo falla al clasificar una instancia.
     */
    private static void generatePredictions(Instances testData, weka.classifiers.Classifier model, String outFile) throws Exception {
        try (BufferedWriter w = new BufferedWriter(new FileWriter(outFile))) {
            for (int i = 0; i < testData.numInstances(); i++) {
                double pred = model.classifyInstance(testData.instance(i));
                String label = testData.classAttribute().value((int) pred);
                w.write(label);
                w.newLine();
            }
        }
    }
}
