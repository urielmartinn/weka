import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import java.util.ArrayList;

/**
 * Software para el Rol de Usuario (Cliente).
 * Carga un modelo pre-entrenado y clasifica nuevos textos crudos desde la terminal.
 */
public class SpamPredictor {

    public static void main(String[] args) {
        // Validación de argumentos por terminal
        if (args.length < 2) {
            System.out.println("Uso incorrecto. El programa requiere ejecutarse sin interacción del usuario.");
            System.out.println("Comando esperado: java SpamPredictor <ruta_del_modelo> \"<texto_a_clasificar>\"");
            System.out.println("Ejemplo: java SpamPredictor spam_classifier_final.model \"You have won a lottery prize!\"");
            return;
        }

        String modelPath = args[0];
        String rawText = args[1]; // El texto crudo que inserta el usuario

        try {
            // 1. Cargar el modelo empaquetado (FilteredClassifier)
            Classifier model = (Classifier) SerializationHelper.read(modelPath);

            // 2. Recrear la estructura de datos EXACTA que usamos en el entrenamiento
            // Atributo 0: Texto (String)
            ArrayList<Attribute> atts = new ArrayList<>();
            atts.add(new Attribute("text", (ArrayList<String>) null));
            
            // Atributo 1: Clase (Nominal: ham, spam)
            ArrayList<String> classValues = new ArrayList<>();
            classValues.add("ham");
            classValues.add("spam");
            atts.add(new Attribute("class", classValues));

            // Crear el dataset vacío
            Instances dataset = new Instances("EmailsPrediction", atts, 1);
            dataset.setClassIndex(dataset.numAttributes() - 1);

            // 3. Crear una nueva instancia con el texto del cliente
            DenseInstance newInstance = new DenseInstance(2);
            // Aplicamos una limpieza básica rápida (la misma que en train)
            String cleanText = cleanTextForPrediction(rawText);
            
            newInstance.setValue(atts.get(0), cleanText);
            // La clase es desconocida (?), por lo que no la seteamos
            newInstance.setDataset(dataset);
            dataset.add(newInstance);

            // 4. Realizar la predicción "on-the-fly"
            // El FilteredClassifier vectorizará el texto automáticamente usando el diccionario guardado
            double predictionIndex = model.classifyInstance(dataset.instance(0));
            String predictedLabel = dataset.classAttribute().value((int) predictionIndex);

            // 5. Mostrar el resultado final al usuario
            System.out.println("--------------------------------------------------");
            System.out.println("Texto analizado: " + rawText);
            System.out.println("PREDICCIÓN: -> " + predictedLabel.toUpperCase() + " <-");
            System.out.println("--------------------------------------------------");

        } catch (Exception e) {
            System.err.println("Error al procesar la predicción: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Mismo método de limpieza que usamos en el entrenamiento para mantener coherencia.
     */
    private static String cleanTextForPrediction(String text) {
        text = text.replaceAll("\\S+@\\S+\\.\\S+", " ");
        text = text.replaceAll("(http|https)://\\S+", " ");
        text = text.replaceAll("\\b\\d+\\b", " ");
        text = text.replaceAll("[^a-zA-Záéíóúñü' ]", " ");
        text = text.toLowerCase();
        text = text.replaceAll("\\s+", " ").trim();
        return text;
    }
}
