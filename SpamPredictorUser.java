import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import java.util.ArrayList;

/**
 * Aplicación de inferencia para el Rol de Usuario (Cliente).
 * <p>
 * Permite cargar un modelo clasificador pre-entrenado (.model) y predecir
 * si un texto introducido por consola es Spam o Ham "on-the-fly".
 * Recrea exactamente la misma estructura vectorial del entrenamiento para
 * evitar fallos de compatibilidad en el diccionario (soluciona el problema Test-blind).
 * </p>
 */
public class SpamPredictorUser {

    /**
     * Método de entrada de la aplicación del cliente.
     * Espera recibir la ruta del modelo y el texto a analizar mediante argumentos de la terminal.
     *
     * @param args Argumentos de terminal. args[0] = path del modelo, args[1] = texto crudo.
     */
    public static void main(String[] args) {
        // [CÓDIGO DEL MAIN]
    }

    /**
     * Aplica la misma limpieza estructural que se usó durante la fase de entrenamiento.
     * Es crucial usar la misma regex para mantener la coherencia semántica antes
     * de pasar el texto al FilteredClassifier.
     *
     * @param text El texto del correo introducido por el cliente.
     * @return El texto en minúsculas sin URLs, emails, números ni puntuaciones complejas.
     */
    private static String cleanTextForPrediction(String text) {
        // [CÓDIGO DE cleanTextForPrediction]
    }
}
