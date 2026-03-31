package Proiektua;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import java.util.ArrayList;

/**
 * Erabiltzailearen aldetik SPAM edo HAM iragarpenak egiteko klasea.
 * <p>
 * Klase honek aldez aurretik entrenatutako Weka eredu bat kargatzen du (.model) eta
 * testu gordin baten iragarpena ("on-the-fly") gauzatzen du terminaletik. Ezinbestekoa
 * da eredu hori FilteredClassifier bidez edota preprozesamendua barnean duela 
 * gorde izana, hiztegien (Test-blind) bateragarritasuna bermatzeko.
 * </p>
 */
public class ErabiltzaileSpamIragarpena {

    /**
     * Programa nagusiaren sarrera-puntua. Eredua kargatu, instantzia berria sortu 
     * eta iragarpena egiten du.
     *
     * @param args Komando-lerroko argumentuak. Bi argumentu behar dira zehazki:
     * args[0]: Entrenatutako ereduaren fitxategiaren ibilbidea (.model).
     * args[1]: Sailkatu nahi den testu gordina (posta elektronikoaren edukia).
     */
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("ERROREA!");
            return;
        }

        String modeloIbilbidea = args[0];
        String testuGordina = args[1]; 

        try {
            
            Classifier model = (Classifier) SerializationHelper.read(modeloIbilbidea);

            ArrayList<Attribute> atts = new ArrayList<>();
            atts.add(new Attribute("text", (ArrayList<String>) null));
            
            ArrayList<String> classValues = new ArrayList<>();
            classValues.add("ham");
            classValues.add("spam");
            atts.add(new Attribute("class", classValues));

            Instances dataset = new Instances("EmailsPrediction", atts, 1);
            dataset.setClassIndex(dataset.numAttributes() - 1);

            DenseInstance InstanceBerria = new DenseInstance(2);
            
            String testuGarbia = cleanTextForPrediction(testuGordina);
            
            InstanceBerria.setValue(atts.get(0), testuGarbia);
            
            InstanceBerria.setDataset(dataset);
            dataset.add(InstanceBerria);

            
            double iragarpenIndizea = model.classifyInstance(dataset.instance(0));
            String iragarpenEtiketa = dataset.classAttribute().value((int) iragarpenIndizea);

            String testuEscapatua = testuGordina.replace("\\\"", "\\\\\"");
            System.out.println(iragarpenEtiketa + ", \"" + testuEscapatua + "\"");

        } catch (Exception e) {
            System.err.println("Iragarpena prozesatzean errorea! " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Iragarpena egin aurretik testua garbitzen eta normalizatzen duen metodo laguntzailea.
     * <p>
     * Hurrengo pausoak aplikatzen ditu adierazpen erregularrak (Regex) erabiliz:
     * 1. Helbide elektronikoak kentzen ditu.
     * 2. URLak kentzen ditu.
     * 3. Zenbakiak kentzen ditu.
     * 4. Alfabetokoak ez diren karaktereak kentzen ditu.
     * 5. Testu osoa minuskuletara pasatzen du.
     * 6. Zuriune bikoitzak edo estrak ezabatzen ditu.
     * </p>
     *
     * @param text Garbitu behar den jatorrizko testu gordina.
     * @return Garbitutako testua, Weka instantzian sartzeko prest.
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
