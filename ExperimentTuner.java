package proiektua;

import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stopwords.Rainbow;
import weka.core.tokenizers.NGramTokenizer;
import weka.core.tokenizers.WordTokenizer;
import weka.classifiers.meta.FilteredClassifier;
import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Clase 1: Encargada de la carga de datos, limpieza, partición (Hold-out) 
 * y optimización de hiperparámetros (Grid Search).
 */
public class ExperimentTuner {

    // ========== ESPACIO DE BÚSQUEDA (Grid Search) ==========
    private static final double[] RIDGE_VALUES = {1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0};
    private static final int[] WORDS_TO_KEEP_VALUES = {100, 250, 500, 1000};
    
    // Diferentes esquemas de vectorización a evaluar empíricamente
    private enum VectorizationType { BINARY, TF, TF_IDF, TF_IDF_BIGRAMS }

    private static final double TRAIN_RATIO = 0.8;
    private static final long RANDOM_SEED = 12345L;

    public static class ExperimentConfig {
        public Instances trainRaw;
        public Instances testRaw;
        public StringToWordVector bestFilter;
        public double bestRidge;
        public int bestWordsToKeep;
        public VectorizationType bestVecType;
        public int finalDictionarySize;
    }

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

    public ExperimentConfig runTuningPipeline(String hamFolder, String spamFolder) throws Exception {
        ExperimentConfig config = new ExperimentConfig();

        // 1. Cargar datos
        System.out.println("Cargando correos...");
        List<Email> allEmails = new ArrayList<>();
        loadEmails(hamFolder, "ham", allEmails);
        loadEmails(spamFolder, "spam", allEmails);

        // 2. Dividir en Train y Test (Hold-out estratificado)
        List<Email> trainEmails = new ArrayList<>();
        List<Email> testEmails = new ArrayList<>();
        splitData(allEmails, trainEmails, testEmails);

        // 3. Crear datasets en crudo para Weka
        config.trainRaw = createInstances(trainEmails, true);
        config.testRaw = createInstances(testEmails, true);

        System.out.println("Iniciando Grid Search (10-fold CV) evaluando Vectorización, WordsToKeep y Ridge...\n");
        System.out.println(String.format("%-15s | %-6s | %-8s | %-9s | %-8s", "Vectorización", "Words", "Ridge", "Dict Size", "F-Measure"));
        System.out.println("-------------------------------------------------------------------------");

        double bestFMeasure = -1.0;
        int spamIndex = config.trainRaw.classAttribute().indexOfValue("spam");

        // 4. GRID SEARCH: Triple bucle anidado
        for (VectorizationType vecType : VectorizationType.values()) {
            for (int currentWords : WORDS_TO_KEEP_VALUES) {
                for (double currentRidge : RIDGE_VALUES) {
                    
                    // A. Configurar el Filtro
                    StringToWordVector tempFilter = new StringToWordVector();
                    tempFilter.setLowerCaseTokens(true);
                    tempFilter.setWordsToKeep(currentWords);
                    tempFilter.setStopwordsHandler(new Rainbow());
                    tempFilter.setStemmer(new LovinsStemmer());

                    // Configurar según el tipo de vectorización a evaluar
                    configureVectorization(tempFilter, vecType);

                    // B. Obtener tamaño exacto del diccionario (Atributos - 1 de la clase)
                    tempFilter.setInputFormat(config.trainRaw);
                    Instances filteredTrain = Filter.useFilter(config.trainRaw, tempFilter);
                    int currentDictSize = filteredTrain.numAttributes() - 1;

                    // C. Configurar el Clasificador Logistic
                    Logistic tempLogistic = new Logistic();
                    tempLogistic.setOptions(new String[]{"-R", String.valueOf(currentRidge), "-M", "500"});

                    // D. Encapsular para SOLUCIONAR EL TEST-BLIND
                    FilteredClassifier tempFc = new FilteredClassifier();
                    tempFc.setFilter(tempFilter);
                    tempFc.setClassifier(tempLogistic);

                    // E. Evaluar con Validación Cruzada sobre el TRAIN
                    Evaluation cvEval = new Evaluation(config.trainRaw);
                    cvEval.crossValidateModel(tempFc, config.trainRaw, 10, new Random(RANDOM_SEED));

                    double fMeasureSpam = cvEval.fMeasure(spamIndex);
                    
                    System.out.println(String.format("%-15s | %6d | %8.1e | %9d | %8.4f", 
                            vecType.name(), currentWords, currentRidge, currentDictSize, fMeasureSpam));

                    // F. Guardar la mejor configuración
                    if (fMeasureSpam > bestFMeasure) {
                        bestFMeasure = fMeasureSpam;
                        config.bestRidge = currentRidge;
                        config.bestWordsToKeep = currentWords;
                        config.bestVecType = vecType;
                        config.finalDictionarySize = currentDictSize;
                        config.bestFilter = tempFilter; 
                    }
                }
            }
        }

        System.out.println("\n=== MEJOR CONFIGURACIÓN ENCONTRADA (Evidencia Empírica) ===");
        System.out.println("Tipo Vectorización: " + config.bestVecType);
        System.out.println("WordsToKeep limit: " + config.bestWordsToKeep);
        System.out.println("Tamaño Real del Diccionario: " + config.finalDictionarySize + " atributos");
        System.out.println("Ridge (Regularización): " + config.bestRidge);
        System.out.println("Mejor F-Measure CV: " + bestFMeasure + "\n");

        return config;
    }

    /**
     * Aplica la configuración matemática y de tokenización según el tipo escogido.
     */
    private void configureVectorization(StringToWordVector filter, VectorizationType type) {
        switch (type) {
            case BINARY:
                filter.setOutputWordCounts(false);
                filter.setTFTransform(false);
                filter.setIDFTransform(false);
                filter.setTokenizer(new WordTokenizer());
                break;
            case TF:
                filter.setOutputWordCounts(true);
                filter.setTFTransform(true);
                filter.setIDFTransform(false);
                filter.setTokenizer(new WordTokenizer());
                break;
            case TF_IDF:
                filter.setOutputWordCounts(true);
                filter.setTFTransform(true);
                filter.setIDFTransform(true);
                filter.setTokenizer(new WordTokenizer());
                break;
            case TF_IDF_BIGRAMS:
                filter.setOutputWordCounts(true);
                filter.setTFTransform(true);
                filter.setIDFTransform(true);
                NGramTokenizer tokenizer = new NGramTokenizer();
                tokenizer.setNGramMinSize(1);
                tokenizer.setNGramMaxSize(2); // Unigramas y Bigramas
                filter.setTokenizer(tokenizer);
                break;
        }
    }

    // =========================================================================
    // MÉTODOS DE UTILIDAD (Carga, Limpieza y Transformación)
    // =========================================================================

    private void loadEmails(String folder, String label, List<Email> list) throws Exception {
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

    private String readFile(File f) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line;
            while ((line = br.readLine()) != null) {
                sb.append(line).append("\n");
            }
        }
        return sb.toString();
    }

    private String cleanText(String text) {
        text = text.replaceAll("\\S+@\\S+\\.\\S+", " ");
        text = text.replaceAll("(http|https)://\\S+", " ");
        text = text.replaceAll("\\b\\d+\\b", " ");
        text = text.replaceAll("[^a-zA-Záéíóúñü' ]", " ");
        text = text.toLowerCase();
        return text.replaceAll("\\s+", " ").trim();
    }

    private void splitData(List<Email> all, List<Email> train, List<Email> test) {
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

    private Instances createInstances(List<Email> emails, boolean withClass) {
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
        data.setClassIndex(data.numAttributes() - 1);

        for (Email e : emails) {
            double[] vals = new double[data.numAttributes()];
            vals[0] = data.attribute(0).addStringValue(e.text);
            vals[1] = withClass ? data.attribute(1).indexOfValue(e.label) : 0;
            data.add(new DenseInstance(1.0, vals));
        }
        return data;
    }
}
