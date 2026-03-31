package proiektua;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Clase 2: Ejecutable independiente para la estimación empírica (Hold-out).
 * Argumentos CLI: <ham_folder> <spam_folder> <output_report.txt>
 */
public class QualityEstimator {

    private static final double TRAIN_RATIO = 0.8;
    private static final long RANDOM_SEED = 12345L;

    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("Uso: java QualityEstimator <ruta_carpeta_ham>");
            System.exit(1);
        }

        String hamFolder = args[0];
        String spamFolder = args[1];
        String reportPath = "estimazioa.txt"; // Ruta del txt de salida

        double bestRidge = 10.0;
        int bestWordsToKeep = 1000;

        try {
            System.out.println("1. Cargando y limpiando correos...");
            List<String[]> allEmails = new ArrayList<>();
            loadEmails(hamFolder, "ham", allEmails);
            loadEmails(spamFolder, "spam", allEmails);

            System.out.println("2. Realizando partición Hold-out (80/20)...");
            List<String[]> trainList = new ArrayList<>();
            List<String[]> testList = new ArrayList<>();
            splitData(allEmails, trainList, testList);

            Instances trainData = createInstances(trainList);
            Instances testData = createInstances(testList);

            System.out.println("3. Solicitando filtro centralizado a NLPFilterFactory...");
            FilteredClassifier fc = new FilteredClassifier();
            fc.setFilter(NLPFilterFactory.createSpamFilter(bestWordsToKeep)); // Llamada a la nueva clase

            Logistic logistic = new Logistic();
            logistic.setOptions(new String[]{"-R", String.valueOf(bestRidge), "-M", "500"});
            fc.setClassifier(logistic);

            System.out.println("4. Entrenando modelo para evaluación...");
            fc.buildClassifier(trainData);

            System.out.println("5. Evaluando sobre el conjunto de Test...");
            Evaluation eval = new Evaluation(trainData);
            eval.evaluateModel(fc, testData); 

            System.out.println("6. Guardando estimación de calidad en disco...");
            saveQualityReport(reportPath, eval, bestRidge, bestWordsToKeep);
            System.out.println("✅ Reporte guardado con éxito en: " + reportPath);

        } catch (Exception e) {
            System.err.println("❌ Error en la evaluación: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void saveQualityReport(String filename, Evaluation eval, double ridge, int words) throws Exception {
        try (BufferedWriter w = new BufferedWriter(new FileWriter(filename))) {
            w.write("=== ESTIMACIÓN DE CALIDAD DEL CLASIFICADOR DE SPAM ===\n");
            w.write("Hiperparámetros óptimos usados: Ridge=" + ridge + ", WordsToKeep=" + words + ", TF-IDF\n\n");
            w.write("Accuracy: " + eval.pctCorrect() + "%\n");
            w.write("Kappa: " + eval.kappa() + "\n");
            w.write("F-Measure ponderado: " + eval.weightedFMeasure() + "\n");
            w.write("\n=== Matriz de confusión ===\n");
            w.write(eval.toMatrixString());
            w.write("\n\n=== Métricas por clase ===\n");
            w.write("Clase ham:\n  Precision: " + eval.precision(0) + "\n  Recall: " + eval.recall(0) + "\n  F-Measure: " + eval.fMeasure(0) + "\n");
            w.write("Clase spam:\n  Precision: " + eval.precision(1) + "\n  Recall: " + eval.recall(1) + "\n  F-Measure: " + eval.fMeasure(1) + "\n");
        }
    }

    // --- MÉTODOS DE UTILIDAD PARA CARGA Y LIMPIEZA ---
    private static void loadEmails(String folder, String label, List<String[]> list) throws Exception {
        File dir = new File(folder);
        if (!dir.exists() || !dir.isDirectory()) return;
        File[] files = dir.listFiles((d, name) -> name.endsWith(".txt"));
        if (files == null) return;
        for (File f : files) {
            StringBuilder sb = new StringBuilder();
            try (BufferedReader br = new BufferedReader(new FileReader(f))) {
                String line;
                while ((line = br.readLine()) != null) sb.append(line).append(" ");
            }
            list.add(new String[]{cleanText(sb.toString()), label});
        }
    }

    private static String cleanText(String text) {
        return text.replaceAll("\\S+@\\S+\\.\\S+", " ").replaceAll("(http|https)://\\S+", " ")
                   .replaceAll("\\b\\d+\\b", " ").replaceAll("[^a-zA-Záéíóúñü' ]", " ")
                   .toLowerCase().replaceAll("\\s+", " ").trim();
    }

    private static void splitData(List<String[]> all, List<String[]> train, List<String[]> test) {
        List<String[]> hamList = new ArrayList<>();
        List<String[]> spamList = new ArrayList<>();
        for (String[] e : all) {
            if (e[1].equals("ham")) hamList.add(e);
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
    }

    private static Instances createInstances(List<String[]> emails) {
        ArrayList<Attribute> atts = new ArrayList<>();
        atts.add(new Attribute("text", (ArrayList<String>) null));
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("ham"); classValues.add("spam");
        atts.add(new Attribute("class", classValues));
        
        Instances data = new Instances("emails", atts, emails.size());
        data.setClassIndex(1);
        for (String[] e : emails) {
            double[] vals = new double[2];
            vals[0] = data.attribute(0).addStringValue(e[0]);
            vals[1] = data.attribute(1).indexOfValue(e[1]);
            data.add(new DenseInstance(1.0, vals));
        }
        return data;
    }
}
