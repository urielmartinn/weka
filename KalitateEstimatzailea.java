package Proiektua;

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
 * SPAM sailkatzailearen kalitatea estimatzeko eta ebaluatzeko klase nagusia.
 * <p>
 * Klase honek datu-multzo osoa kargatzen du, Hold-out (%80 Train / %20 Test) 
 * partizio estratifikatua egiten du eta Logistic Regression modeloa entrenatzen 
 * zein ebaluatzen du (Test-blind printzipioa errespetatuz FilteredClassifier bidez). 
 * Azkenik, ebaluazioaren metrika nagusiak testu-fitxategi batean gordetzen ditu.
 * </p>
 */
public class KalitateEstimatzailea {

    /** Entrenamendu-partizioaren tamaina (%80). */
    private static final double TRAIN_RATIOA = 0.8;
    
    /** Partizioak egiterakoan ausazkotasuna kontrolatzeko hazia (erreproduzigarritasuna bermatzeko). */
    private static final long AUSAZKO_HAZIA = 12345L;

    /**
     * Programa nagusiaren sarrera-puntua.
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
        String txostenIbilbidea = "estimazioa.txt";

        double RidgeOnena = 10.0;
        int wordsToKeepOnena = 1000;

        try {
            System.out.println("1. Mail-ak kargatzen eta garbitzen...");
            List<String[]> allEmails = new ArrayList<>();
            emailakKargatu(hamKarpeta, "ham", allEmails);
            emailakKargatu(spamKarpeta, "spam", allEmails);

            System.out.println("2. Hold-out partizioa egiten (80/20)...");
            List<String[]> trainList = new ArrayList<>();
            List<String[]> testList = new ArrayList<>();
            splitData(allEmails, trainList, testList);

            Instances trainData = createInstances(trainList);
            Instances testData = createInstances(testList);

            System.out.println("3. Iragazki zentralizatua eskatuz IragazkiSortzaileari...");
            FilteredClassifier fC = new FilteredClassifier();
            fC.setFilter(IragazkiSortzailea.sortuSpamIragazkia(wordsToKeepOnena));

            Logistic logistic = new Logistic();
            logistic.setOptions(new String[]{"-R", String.valueOf(RidgeOnena), "-M", "500"});
            fC.setClassifier(logistic);

            System.out.println("4. Ebaluatzeko eredua entrenatzen...");
            fC.buildClassifier(trainData);

            System.out.println("5. Test guztien gainean ebaluatuz...");
            Evaluation ebaluazioa = new Evaluation(trainData);
            ebaluazioa.evaluateModel(fC, testData); 

            System.out.println("6. diskoko kalitate-estimazioa gordetzen...");
            gordeKalitateTxostena(txostenIbilbidea, ebaluazioa, RidgeOnena, wordsToKeepOnena);
            System.out.println("txostena hemen gordeta:" + txostenIbilbidea);

        } catch (Exception e) {
            System.err.println("Errorea ebaluazioan: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Ebaluazioaren emaitzak eta metrikak testu-fitxategi batean idazten ditu.
     *
     * @param artxiboIzena Sortuko den txostenaren fitxategi-izena.
     * @param ebaluazioa2  Weka-ko Evaluation objektua, ebaluazioaren emaitzak dituena.
     * @param ridge        Erabilitako Ridge (erregularizazioa) hiperparametroaren balioa.
     * @param hitzak       Hiztegian mantendu diren hitz kopurua (wordsToKeep).
     * @throws Exception Fitxategia idazterakoan arazoren bat badago.
     */
    private static void gordeKalitateTxostena(String artxiboIzena, Evaluation ebaluazioa2, double ridge, int hitzak) throws Exception {
        try (BufferedWriter w = new BufferedWriter(new FileWriter(artxiboIzena))) {
            w.write("=== SPAM SAILKATZAILEAREN KALITATE-ESTIMAZIOA ===\n");
            w.write("Hiperparametro optimo erabiliak: Ridge=" + ridge + ", WordsToKeep=" + hitzak + ", TF-IDF\n\n");
            w.write("Zehaztapena: " + ebaluazioa2.pctCorrect() + "%\n");
            w.write("Kappa: " + ebaluazioa2.kappa() + "\n");
            w.write("F-Measure metatua: " + ebaluazioa2.weightedFMeasure() + "\n");
            w.write("\n=== Nahasmen matrizea ===\n");
            w.write(ebaluazioa2.toMatrixString());
            w.write("\n\n=== Klaseko metrikak ===\n");
            w.write("ham klasea:\n  Zehaztapena: " + ebaluazioa2.precision(0) + "\n  Recall: " + ebaluazioa2.recall(0) + "\n  F-Measure: " + ebaluazioa2.fMeasure(0) + "\n");
            w.write("spam klasea:\n  Zehaztapena: " + ebaluazioa2.precision(1) + "\n  Recall: " + ebaluazioa2.recall(1) + "\n  F-Measure: " + ebaluazioa2.fMeasure(1) + "\n");
        }
    }

    /**
     * Karpeta zehatz bateko .txt fitxategi guztiak irakurtzen ditu, testua garbitzen du
     * eta zerrenda batean gordetzen ditu emandako etiketarekin batera.
     *
     * @param karpeta Mezuen testu-fitxategiak dituen direktorioaren ibilbidea.
     * @param etiketa Klasearen etiketa ("ham" edo "spam").
     * @param list    Garbitutako mezuak eta haien etiketak gordeko diren zerrenda.
     * @throws Exception Fitxategiak irakurtzerakoan arazoren bat badago.
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
     * Testu gordina garbitzen du iragarpenak egiteko edo entrenatzeko prest egoteko.
     * Adierazpen erregularrak (Regex) erabiltzen ditu zaborra (URLak, e-mailak, zenbakiak...) ezabatzeko.
     *
     * @param testua Garbitu behar den jatorrizko testua.
     * @return Testu garbia eta normalizatua (minuskulatan).
     */
    private static String testuaGarbitu(String testua) {
        return testua.replaceAll("\\S+@\\S+\\.\\S+", " ").replaceAll("(http|https)://\\S+", " ")
                   .replaceAll("\\b\\d+\\b", " ").replaceAll("[^a-zA-Záéíóúñü' ]", " ")
                   .toLowerCase().replaceAll("\\s+", " ").trim();
    }

    /**
     * Datu-multzo osoa bi partiziotan banatzen du: Entrenamendua (Train) eta Testa (Test).
     * Banaketa estratifikatua da, hau da, 'ham' eta 'spam' proportzioak mantentzen ditu.
     *
     * @param dena  Datu-multzo osoa (mezu garbiak eta etiketak).
     * @param train Entrenamendu-datuak gordeko diren zerrenda.
     * @param test  Test-datuak (ebaluaziorako) gordeko diren zerrenda.
     */
    private static void splitData(List<String[]> dena, List<String[]> train, List<String[]> test) {
        List<String[]> hamList = new ArrayList<>();
        List<String[]> spamList = new ArrayList<>();
        for (String[] e : dena) {
            if (e[1].equals("ham")) hamList.add(e);
            else spamList.add(e);
        }
        Collections.shuffle(hamList, new Random(AUSAZKO_HAZIA));
        Collections.shuffle(spamList, new Random(AUSAZKO_HAZIA));
        
        int hamTrainSize = (int) Math.round(hamList.size() * TRAIN_RATIOA);
        int spamTrainSize = (int) Math.round(spamList.size() * TRAIN_RATIOA);
        
        train.addAll(hamList.subList(0, hamTrainSize));
        train.addAll(spamList.subList(0, spamTrainSize));
        test.addAll(hamList.subList(hamTrainSize, hamList.size()));
        test.addAll(spamList.subList(spamTrainSize, spamList.size()));
    }

    /**
     * Karaktere-kateen zerrenda bat Weka liburutegiak onartzen duen {@link Instances} 
     * objektu batean bihurtzen du.
     *
     * @param emailak Mezu garbien eta haien etiketen zerrenda.
     * @return Weka-rako prestatutako 'Instances' datu-multzoa.
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
            double[] balioa = new double[2];
            balioa[0] = data.attribute(0).addStringValue(e[0]);
            balioa[1] = data.attribute(1).indexOfValue(e[1]);
            data.add(new DenseInstance(1.0, balioa));
        }
        return data;
    }
}
