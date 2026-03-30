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
import java.util.Random; //No es necesario

/**
 * Clasificador de spam con regresión logística (parámetros fijos).
 * Lee correos de las carpetas "ham" y "spam", divide en train/test,
 * entrena y evalúa, genera summary, calidad y predicciones.
 */
public class SpamClassifierAdmin {

    // ========== PARÁMETROS FIJOS (cámbialos aquí si quieres) ==========
    private static final double[] RIDGE_VALUES = {1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0};  // parámetro de regularización
    private static final int WORDS_TO_KEEP = 1000;     // número de palabras a mantener
    //private static final boolean USE_TFIDF = true;     // true = TF-IDF, false = frecuencias
    private static final double TRAIN_RATIO = 0.8;     // 80% entrenamiento, 20% test
    private static final long RANDOM_SEED = 12345L;    // para reproducibilidad
    //Aurreprosezamendua
    private static final boolean USE_STOPWORDS = true; // true = elimina palabras vacías
    private static final boolean USE_STEMMER = true;   // true = reduce a la raíz (lematización)
    //Bektorizazioa
    private static final boolean OUTPUT_WORD_COUNTS = true; // true = frecuencias, false = binario
    private static final boolean USE_TF = true;             // true = aplica logaritmo a las frecuencias
    private static final boolean USE_IDF = true;            // true = penaliza palabras comunes
    private static final boolean USE_BIGRAMS = false;       // true = unigramas y bigramas (ej. "free", "free money")
    
    // ========== CLASE AUXILIAR PARA CORREOS ==========
    private static class Email {
        String text;
        String label;
        String author;
        Date date;

        Email(String text, String label, String author, Date date) {
            this.text = text;
            this.label = label;
            this.author = author;
            this.date = date;
        }
    }

    // ========== MAIN ==========
    public static void main(String[] args) throws Exception {
        // Leer carpetas desde argumentos o usar valores por defecto
        String hamFolder = (args.length > 0) ? args[0] : "ham";
        String spamFolder = (args.length > 1) ? args[1] : "spam";
        String outputSummary = "summary.txt";
        String outputQuality = "quality.txt";
        String outputPredictions = "predictions.txt";

        // 1. Cargar todos los correos
        System.out.println("Cargando correos desde " + hamFolder + " y " + spamFolder + "...");
        List<Email> allEmails = new ArrayList<>();
        loadEmails(hamFolder, "ham", allEmails);
        loadEmails(spamFolder, "spam", allEmails);
        System.out.println("Total correos cargados: " + allEmails.size());

        // 2. Generar resumen
        System.out.println("Generando resumen...");
        generateSummary(allEmails, outputSummary);

        // 3. Dividir en entrenamiento y test
        System.out.println("Dividiendo en train/test (ratio=" + TRAIN_RATIO + ")...");
        List<Email> trainEmails = new ArrayList<>();
        List<Email> testEmails = new ArrayList<>();
        splitData(allEmails, trainEmails, testEmails);
        System.out.println("Entrenamiento: " + trainEmails.size() + " correos");
        System.out.println("Test: " + testEmails.size() + " correos");

        // 4. Crear datasets Weka (texto crudo)
        Instances trainRaw = createInstances(trainEmails, true);
        Instances testRaw = createInstances(testEmails, true); // con etiquetas para evaluación
        trainRaw.setClassIndex(trainRaw.numAttributes() - 1);
        testRaw.setClassIndex(testRaw.numAttributes() - 1);

        // 5. Configurar filtro de vectorización
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

        // 6. Entrenar regresión logística con FILTERED CLASSIFIER (¡CRÍTICO!)
        
        System.out.println("RIDGE");
        System.out.println("10-fold Cross-Validation en Train");
        
        double bestRidge = RIDGE_VALUES[0];
        double bestFMeasure = -1.0;

        // Bucle iterativo para probar cada valor
        for (double currentRidge : RIDGE_VALUES) {
            Logistic tempLogistic = new Logistic();
            tempLogistic.setOptions(new String[]{"-R", String.valueOf(currentRidge), "-M", "500"});

            FilteredClassifier tempFc = new FilteredClassifier();
            tempFc.setFilter(filter); // Usamos el filtro configurado en el Paso 5
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
        

        // 6b. Entrenar el modelo FINAL con el MEJOR Ridge usando TODOS los datos de Train
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
        System.out.println(" - " + outputSummary);
        System.out.println(" - " + outputQuality);
        System.out.println(" - " + outputPredictions);
        System.out.println(" - spam_classifier_final.model (¡Listo para el cliente!)");
    }
    // ==================== CARGA DE CORREOS ====================
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

    // ==================== GENERAR SUMMARY ====================
    private static void generateSummary(List<Email> emails, String outFile) throws IOException {
        List<Email> ham = new ArrayList<>();
        List<Email> spam = new ArrayList<>();
        for (Email e : emails) {
            if (e.label.equals("ham")) ham.add(e);
            else spam.add(e);
        }

        ham.sort(Comparator.comparing(e -> e.date, Comparator.nullsLast(Comparator.naturalOrder())));
        spam.sort(Comparator.comparing(e -> e.date, Comparator.nullsLast(Comparator.naturalOrder())));

        String authorHam = mostFrequentAuthor(ham);
        String authorSpam = mostFrequentAuthor(spam);
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");

        try (BufferedWriter w = new BufferedWriter(new FileWriter(outFile))) {
            w.write("Legitimate\n");
            w.write("----------\n");
            w.write("- Owner: " + authorHam + "\n");
            w.write("- Total number: " + ham.size() + " emails\n");
            w.write("- Date of first email: " + (ham.isEmpty() ? "N/A" : sdf.format(ham.get(0).date)) + "\n");
            w.write("- Date of last email: " + (ham.isEmpty() ? "N/A" : sdf.format(ham.get(ham.size()-1).date)) + "\n");
            w.write("- Similars deletion: No\n");
            w.write("- Encoding: No\n");
            w.write("\n\n");
            w.write("Spam\n");
            w.write("----\n");
            w.write("- Owner: " + authorSpam + "\n");
            w.write("- Total number: " + spam.size() + " emails\n");
            w.write("- Date of first email: " + (spam.isEmpty() ? "N/A" : sdf.format(spam.get(0).date)) + "\n");
            w.write("- Date of last email: " + (spam.isEmpty() ? "N/A" : sdf.format(spam.get(spam.size()-1).date)) + "\n");
            w.write("- Similars deletion: No\n");
            w.write("- Encoding: No\n");
            w.write("\n");
            int ratio = (spam.size() == 0) ? 0 : (int) Math.round((double) ham.size() / spam.size());
            w.write("Spam:Legitimate rate = 1:" + ratio + "\n");
            w.write("Total number of emails (legitimate + spam): " + emails.size() + "\n");
        }
    }

    private static String mostFrequentAuthor(List<Email> emails) {
        if (emails.isEmpty()) return "desconocido";
        Map<String, Integer> freq = new HashMap<>();
        for (Email e : emails) {
            if (e.author != null)
                freq.put(e.author, freq.getOrDefault(e.author, 0) + 1);
        }
        return freq.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("desconocido");
    }

    // ==================== DIVISIÓN TRAIN/TEST ====================
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

    // ==================== CREAR INSTANCES W EKA ====================
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

    // ==================== GUARDAR INFORME DE CALIDAD ====================
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

    // ==================== GENERAR PREDICCIONES ====================
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
