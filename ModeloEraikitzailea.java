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
 * Produkziorako SPAM sailkatzailearen eredu finala eraikitzeko eta gordetzeko klasea.
 * <p>
 * Klase honek datu-multzo osoa (entrenamendu zein test partizioak bereizi gabe)
 * erabiltzen du eredua entrenatzeko. Parametro optimoak aplikatzen ditu 
 * (Ridge eta wordsToKeep) eta entrenatutako eredua diskoan serializatzen du 
 * (.model formatuan), geroago bezeroaren aldetik iragarpenak egiteko erabil dadin.
 * </p>
 */
public class ModeloEraikitzailea {

    /**
     * Programa nagusiaren sarrera-puntua. Mezu guztiak kargatzen ditu, iragazkia 
     * eta sailkatzailea (Logistic) konfiguratzen ditu, eredua entrenatzen du eta,
     * azkenik, diskoan gordetzen du.
     *
     * @param args Komando-lerroko argumentuak. Bi argumentu behar dira:
     * args[0]: 'ham' mezuak dituen karpetaren ibilbidea.
     * args[1]: 'spam' mezuak dituen karpetaren ibilbidea.
     */
    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("Errorea!");
            System.exit(1);
        }

        String hamKarpeta = args[0];
        String spamKarpeta = args[1];
        String modelOutputPath = "modeloa.model";

        double RidgeOnena = 10.0;
        int WordsToKeepOnena = 1000;

        try {
            System.out.println("1. mail-ak kargatzen produkziorako...");
            List<String[]> emailGuztiak = new ArrayList<>();
            emailakKargatu(hamKarpeta, "ham", emailGuztiak);
            emailakKargatu(spamKarpeta, "spam", emailGuztiak);

            Instances dataGuztia = createInstances(emailGuztiak);

            System.out.println("2. Iragazki zentralizatua eskatuz IragazkiSortzaileari...");
            FilteredClassifier modelFinala = new FilteredClassifier();
            modelFinala.setFilter(IragazkiSortzailea.sortuSpamIragazkia(WordsToKeepOnena));

            Logistic logisticFinala = new Logistic();
            logisticFinala.setOptions(new String[]{"-R", String.valueOf(RidgeOnena), "-M", "500"});
            modelFinala.setClassifier(logisticFinala);

            System.out.println("3. Modelo finala entrenatzen honekin:"
                    + " " + dataGuztia.numInstances() + " instantziak...");
            modelFinala.buildClassifier(dataGuztia);

            System.out.println("4. Modeloa serializatzen...");
            SerializationHelper.write(modelOutputPath, modelFinala);
            
            System.out.println("Arrakastaz serializatuta eredua hemen: " + modelOutputPath);

        } catch (Exception e) {
            System.err.println("Errorea modeloa egiten: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Karpeta zehatz bateko .txt fitxategi guztiak irakurtzen ditu, testua garbitzen du
     * eta emandako etiketa erabiliz zerrenda batera gehitzen ditu.
     *
     * @param karpeta Mezuen testu-fitxategiak dituen direktorioaren ibilbidea.
     * @param etiketa Klasearen etiketa ("ham" edo "spam").
     * @param list    Garbitutako mezuak (eta euren etiketak) gordeko diren zerrenda.
     * @throws Exception Fitxategiak irakurtzerakoan arazoren bat gertatzen bada.
     */
    private static void emailakKargatu(String karpeta, String etiketa, List<String[]> list) throws Exception {
        File ibilbidea = new File(karpeta);
        if (!ibilbidea.exists() || !ibilbidea.isDirectory()) return;
        File[] artxiboak = ibilbidea.listFiles((d, name) -> name.endsWith(".txt"));
        if (artxiboak == null) return;
        for (File f : artxiboak) {
            StringBuilder sB = new StringBuilder();
            try (BufferedReader bR = new BufferedReader(new FileReader(f))) {
                String line;
                while ((line = bR.readLine()) != null) sB.append(line).append(" ");
            }
            list.add(new String[]{testuaGarbitu(sB.toString()), etiketa});
        }
    }

    /**
     * Testu gordina garbitzen du, adierazpen erregularrak (Regex) erabiliz 
     * informazio ez-garrantzitsua kentzeko (e-mailak, URLak, zenbakiak, etab.)
     * eta testu osoa minuskuletara pasatzeko.
     *
     * @param text Garbitu beharreko jatorrizko testu gordina.
     * @return Testu garbia eta normalizatua.
     */
    private static String testuaGarbitu(String text) {
        return text.replaceAll("\\S+@\\S+\\.\\S+", " ").replaceAll("(http|https)://\\S+", " ")
                   .replaceAll("\\b\\d+\\b", " ").replaceAll("[^a-zA-Záéíóúñü' ]", " ")
                   .toLowerCase().replaceAll("\\s+", " ").trim();
    }

    /**
     * Karaktere-kateen zerrenda bat (testua + etiketa) Weka liburutegiak onartzen duen 
     * {@link Instances} objektu batean eraldatzen du.
     *
     * @param emailak Garbitutako mezuak eta haien etiketak dituen zerrenda.
     * @return Eredua entrenatzeko prest dagoen Weka 'Instances' objektua.
     */
    private static Instances createInstances(List<String[]> emailak) {
        ArrayList<Attribute> atts = new ArrayList<>();
        atts.add(new Attribute("text", (ArrayList<String>) null));
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("ham"); classValues.add("spam");
        atts.add(new Attribute("class", classValues));
        
        Instances data = new Instances("emailak", atts, emailak.size());
        data.setClassIndex(1);
        for (String[] e : emailak) {
            double[] balioak = new double[2];
            balioak[0] = data.attribute(0).addStringValue(e[0]);
            balioak[1] = data.attribute(1).indexOfValue(e[1]);
            data.add(new DenseInstance(1.0, balioak));
        }
        return data;
    }
}
