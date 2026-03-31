package Proiektua;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import java.util.ArrayList;


public class SpamPredictorUser {

   
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

           
            System.out.println("--------------------------------------------------");
            System.out.println("Aztertutako testua: " + testuGordina);
            System.out.println("IRAGARPENA: -> " + iragarpenEtiketa.toUpperCase() + " <-");
            System.out.println("--------------------------------------------------");

        } catch (Exception e) {
            System.err.println("Iragarpena prozesatzean errorea! " + e.getMessage());
            e.printStackTrace();
        }
    }

    
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

