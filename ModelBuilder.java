package proiektua;

import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Clase 3: Ejecutable independiente para empaquetar el modelo de producción.
 * Argumentos CLI: <ham_folder> <spam_folder>
 */
public class ModelBuilder {

    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("Uso: java ModelBuilder <ruta_carpeta_ham>");
            System.exit(1);
        }

        String hamFolder = args[0];
        String spamFolder = args[1];
        String modelOutputPath = "modeloa.model";

        double bestRidge = 10.0;
        int bestWordsToKeep = 1000;

        try {
            System.out.println("1. Cargando EL 100% de los correos para producción...");
            List<String[]> allEmails = new ArrayList<>();
            loadEmails(hamFolder, "ham", allEmails);
            loadEmails(spamFolder, "spam", allEmails);

            Instances allData = createInstances(allEmails);

            System.out.println("2. Solicitando filtro centralizado a NLPFilterFactory...");
            FilteredClassifier finalModel = new FilteredClassifier();
            finalModel.setFilter(NLPFilterFactory.createSpamFilter(bestWordsToKeep)); // Llamada a la nueva clase

            Logistic finalLogistic = new Logistic();
            finalLogistic.setOptions(new String[]{"-R", String.valueOf(bestRidge), "-M", "500"});
            finalModel.setClassifier(finalLogistic);

            System.out.println("3. Entrenando el modelo final con " + allData.numInstances() + " instancias...");
            finalModel.buildClassifier(allData);

            System.out.println("4. Serializando el modelo...");
            SerializationHelper.write(modelOutputPath, finalModel);
            
            System.out.println("✅ Modelo serializado con éxito en: " + modelOutputPath);

        } catch (Exception e) {
            System.err.println("❌ Error construyendo el modelo: " + e.getMessage());
            e.printStackTrace();
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
