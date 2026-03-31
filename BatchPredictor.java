package proiektua;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;

/**
 * Clase 4: Genera un listado de predicciones para los correos de las carpetas de datos.
 * Formato de salida: Prediccion, "mail"
 */
public class BatchPredictor {

    public static void main(String[] args) {
        if (args.length != 4) {
            System.err.println("Uso: java BatchPredictor <folder_ham> <folder_spam> <modelo.model> <predicciones.txt>");
            System.exit(1);
        }

        String hamFolder = args[0];
        String spamFolder = args[1];
        String modelPath = args[2];
        String outputPath = args[3];

        try {
            System.out.println("Cargando modelo...");
            Classifier model = (Classifier) SerializationHelper.read(modelPath);
            Instances dataStructure = createDataStructure();

            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {
                // No escribo cabecera para ceñirme estrictamente a tu formato: Prediccion, "mail"
                
                // Procesar ambas carpetas
                processFolder(new File(hamFolder), model, dataStructure, writer);
                processFolder(new File(spamFolder), model, dataStructure, writer);
            }

            System.out.println("✅ Predicciones exportadas a: " + outputPath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void processFolder(File folder, Classifier model, Instances ds, BufferedWriter writer) throws Exception {
        if (!folder.exists() || !folder.isDirectory()) return;
        File[] files = folder.listFiles((d, name) -> name.endsWith(".txt"));
        if (files == null) return;

        for (File f : files) {
            String text = cleanText(readFile(f));
            
            // Crear instancia compatible con el modelo
            DenseInstance inst = new DenseInstance(2);
            inst.setDataset(ds);
            inst.setValue(0, text);
            inst.setMissing(1);

            // Predecir índice de la clase
            double pred = model.classifyInstance(inst);
            String label = ds.classAttribute().value((int) pred);

            // Formato: Prediccion, "mail"
            writer.write(label + ", \"" + f.getName() + "\"\n");
        }
    }

    private static Instances createDataStructure() {
        ArrayList<Attribute> atts = new ArrayList<>();
        atts.add(new Attribute("text", (ArrayList<String>) null));
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("ham");
        classValues.add("spam");
        atts.add(new Attribute("class", classValues));
        Instances ds = new Instances("Inference", atts, 0);
        ds.setClassIndex(1);
        return ds;
    }

    private static String readFile(File f) throws Exception {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line;
            while ((line = br.readLine()) != null) sb.append(line).append(" ");
        }
        return sb.toString();
    }

    private static String cleanText(String text) {
        return text.replaceAll("\\S+@\\S+\\.\\S+", " ").replaceAll("(http|https)://\\S+", " ")
                   .replaceAll("\\b\\d+\\b", " ").replaceAll("[^a-zA-Záéíóúñü' ]", " ")
                   .toLowerCase().replaceAll("\\s+", " ").trim();
    }
}
