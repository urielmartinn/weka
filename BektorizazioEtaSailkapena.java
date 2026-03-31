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
 * Klase honek esperimentazioa kudeatzen du, bektorizazio mota,
 * hiztegiaren tamaina (WordsToKeep) eta erregresio logistikoaren
 * ridge parametroa optimizatzeko.
 * <p>
 * Grid Search erabiliz, 10-fold cross-validation egiten du eta
 * F‑Measure (spam) metrika hobetzen duen konfigurazioa bilatzen du.
 * </p>
 * 
 */
public class BektorizazioEtaSailkapena {

    // ---------- Parametro espazioa (Grid Search) ----------
    /** Probarikoko ridge balioak (L2 erregulazioa). */
    private static final double[] RIDGE_BALIOAK = {1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0};
    /** Hiztegian mantenduko diren hitz kopuru posibleak. */
    private static final int[] WORDS_TO_KEEP_BALIOAK = {100, 250, 500, 1000};
    
    /** Bektorizazio mota ezberdinak: binarioa, TF, TF‑IDF eta TF‑IDF bigramaekin. */
    private enum BektorizazioMota { BINARY, TF, TF_IDF, TF_IDF_BIGRAMS }

    /** Trebakuntzarako datuen portzentajea (gainontzekoa test). */
    private static final double TRAIN_RATIO = 0.8;
    /** Aleatoriotasuna errepikagarria izateko hazi finkoa. */
    private static final long RANDOM_HAZIA = 12345L;

    /**
     * Konfigurazio optimoaren emaitzak biltzen dituen klase laguntzailea.
     * <p>
     * Atributuek trebakuntza eta test instantziak, hiztegiaren tamaina,
     * ridge balioa eta aukeratutako bektorizazio mota gordetzen dute.
     * </p>
     */
    public static class ExperimentoKonfigurazioa {
        public Instances trainRaw;
        public Instances testRaw;
        public StringToWordVector iragazkiOnena;
        public double ridgeOnena;
        public int wordsToKeepOnena;
        public BektorizazioMota bektorizazioMotaOnena;
        public int hiztegiHitzKopurua;
    }

    /**
     * Posta elektroniko baten informazioa gordetzeko klase laguntzailea.
     */
    private static class Email {
        String testua;        // mezuaren edukia
        String mota;       // "ham" edo "spam"
        String egilea;      // fitxategi-izenetik ateratako egilea
        Date date;          // fitxategi-izenetik ateratako data

        Email(String testua, String mota, String egilea, Date date) {
            this.testua = testua;
            this.mota = mota;
            this.egilea = egilea;
            this.date = date;
        }
    }

    /**
     * Esperimentu osoa exekutatzen du: datuak kargatu, partitu, bektorizazio
     * mota, hitz kopurua eta ridge parametroa aztertu, eta konfigurazio
     * optimoa itzuli.
     *
     * @param hamDirektorioa  ham mezuak dituen karpeta
     * @param spamDirektorioa spam mezuak dituen karpeta
     * @return ExperimentoKonfigurazioa objektua, konfigurazio onenarekin
     * @throws Exception irakurketa, filtro edo ebaluazio erroreak gertatuz gero
     */
    public ExperimentoKonfigurazioa runTuningPipeline(String hamDirektorioa, String spamDirektorioa ) throws Exception {
    	ExperimentoKonfigurazioa konfigurazioa = new ExperimentoKonfigurazioa();

        System.out.println("Posta elektronikoak kargatzen...");
        List<Email> emailGuztiak = new ArrayList<>();
        emailakKargatu(hamDirektorioa, "ham", emailGuztiak);
        emailakKargatu(spamDirektorioa, "spam", emailGuztiak);

        List<Email> trainEmails = new ArrayList<>();
        List<Email> testEmails = new ArrayList<>();
        dataZatitu(emailGuztiak, trainEmails, testEmails);

        konfigurazioa.trainRaw = instantziakSortu(trainEmails, true);
        konfigurazioa.testRaw = instantziakSortu(testEmails, true);

        System.out.println("Grid Search abiatzen (10-fold CV) bektorizazioa, WordsToKeep eta Ridge ebaluatzeko...\n");
        System.out.println(String.format("%-15s | %-6s | %-8s | %-9s | %-8s", "Bektorizazioa", "Hitzak", "Ridge", "Hiztegia", "F-Measure"));
        System.out.println("-------------------------------------------------------------------------");

        double fOnena = -1.0;
        int spamIndex = konfigurazioa.trainRaw.classAttribute().indexOfValue("spam");

        for (BektorizazioMota bekMota : BektorizazioMota.values()) {
            for (int hitzak : WORDS_TO_KEEP_BALIOAK) {
                for (double rOnena : RIDGE_BALIOAK) {
                    
                    StringToWordVector iragazkia = new StringToWordVector();
                    iragazkia.setLowerCaseTokens(true);
                    iragazkia.setWordsToKeep(hitzak);
                    iragazkia.setStopwordsHandler(new Rainbow());
                    iragazkia.setStemmer(new LovinsStemmer());

                    configureVectorization(iragazkia, bekMota);

                    iragazkia.setInputFormat(konfigurazioa.trainRaw);
                    Instances trainIragazita = Filter.useFilter(konfigurazioa.trainRaw, iragazkia);
                    int hitzKop = trainIragazita.numAttributes() - 1;

                    Logistic tempLogistic = new Logistic();
                    tempLogistic.setOptions(new String[]{"-R", String.valueOf(rOnena), "-M", "500"});

                    FilteredClassifier tempFc = new FilteredClassifier();
                    tempFc.setFilter(iragazkia);
                    tempFc.setClassifier(tempLogistic);

                    Evaluation cvEval = new Evaluation(konfigurazioa.trainRaw);
                    cvEval.crossValidateModel(tempFc, konfigurazioa.trainRaw, 10, new Random(RANDOM_HAZIA));

                    double fMeasureSpam = cvEval.fMeasure(spamIndex);
                    
                    System.out.println(String.format("%-15s | %6d | %8.1e | %9d | %8.4f", 
                    		bekMota.name(), hitzak, rOnena, hitzKop, fMeasureSpam));

                    if (fMeasureSpam > fOnena) {
                    	fOnena = fMeasureSpam;
                    	konfigurazioa.ridgeOnena = rOnena;
                    	konfigurazioa.wordsToKeepOnena = hitzak;
                    	konfigurazioa.bektorizazioMotaOnena = bekMota;
                        konfigurazioa.hiztegiHitzKopurua = hitzKop;
                        konfigurazioa.iragazkiOnena = iragazkia; 
                    }
                }
            }
        }

        System.out.println("\n=== AURKITUTAKO KONFIGURAZIO ONENA (Ebidentzia Enpirikoa) ===");
        System.out.println("Bektorizazio mota: " + konfigurazioa.bektorizazioMotaOnena);
        System.out.println("WordsToKeep muga: " + konfigurazioa.wordsToKeepOnena);
        System.out.println("Hiztegiaren benetako tamaina: " + konfigurazioa.hiztegiHitzKopurua + " atributu");
        System.out.println("Ridge (Erregulazioa): " + konfigurazioa.ridgeOnena);
        System.out.println("F-Measure (CV) hoberena: " + fOnena + "\n");

        return konfigurazioa;
    }

    /**
     * StringToWordVector iragazkia konfiguratzen du bektorizazio motaren arabera.
     * <p>
     * Lau kasu: binarioa (presencia/ausencia), TF (maiztasuna), TF‑IDF,
     * eta TF‑IDF unigramak+bigramak erabiliz.
     * </p>
     *
     * @param iragazkia konfiguratu beharreko iragazkia
     * @param mota   bektorizazio mota (BINARY, TF, TF_IDF, TF_IDF_BIGRAMS)
     */
    private void configureVectorization(StringToWordVector iragazkia, BektorizazioMota mota) {
        switch (mota) {
            case BINARY:
            	iragazkia.setOutputWordCounts(false);
            	iragazkia.setTFTransform(false);
            	iragazkia.setIDFTransform(false);
            	iragazkia.setTokenizer(new WordTokenizer());
                break;
            case TF:
            	iragazkia.setOutputWordCounts(true);
            	iragazkia.setTFTransform(true);
            	iragazkia.setIDFTransform(false);
            	iragazkia.setTokenizer(new WordTokenizer());
                break;
            case TF_IDF:
            	iragazkia.setOutputWordCounts(true);
            	iragazkia.setTFTransform(true);
            	iragazkia.setIDFTransform(true);
            	iragazkia.setTokenizer(new WordTokenizer());
                break;
            case TF_IDF_BIGRAMS:
            	iragazkia.setOutputWordCounts(true);
            	iragazkia.setTFTransform(true);
            	iragazkia.setIDFTransform(true);
                NGramTokenizer tokenizer = new NGramTokenizer();
                tokenizer.setNGramMinSize(1);
                tokenizer.setNGramMaxSize(2); 
                iragazkia.setTokenizer(tokenizer);
                break;
        }
    }

    /**
     * Karpeta batetik .txt fitxategi guztiak irakurtzen ditu eta Email objektu
     * gisa gordetzen.
     *
     * @param direktorioa karpetaren bidea
     * @param mota  "ham" edo "spam"
     * @param lista   Email zerrenda non metatuko diren
     * @throws Exception irakurketa erroreak
     */
    private void emailakKargatu(String direktorioa, String mota, List<Email> lista) throws Exception {
        File dir = new File(direktorioa);
        if (!dir.exists() || !dir.isDirectory()) {
            System.err.println("Karpeta ez da existitzen: " + direktorioa);
            return;
        }
        File[] fitxategiak = dir.listFiles((d, izena) -> izena.endsWith(".txt"));
        if (fitxategiak == null) return;

        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
        for (File f : fitxategiak) {
            String edukia = fitxategiaIrakurri(f);
            String izena = f.getName();
            String[] atalak = izena.split("\\.");
            String egilea = (atalak.length >= 3) ? atalak[2] : "ezezaguna";
            Date date = null;
            if (atalak.length >= 2) {
                try { date = sdf.parse(atalak[1]); } catch (Exception e) {}
            }
            edukia = testuaGarbitu(edukia);
            lista.add(new Email(edukia, mota, egilea, date));
        }
    }

    /**
     * Fitxategi bat irakurtzen du eta eduki osoa String gisa itzultzen du.
     *
     * @param f irakurri beharreko fitxategia
     * @return fitxategiaren testua
     * @throws IOException irakurketa errorea
     */
    private String fitxategiaIrakurri(File f) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String lerroa;
            while ((lerroa = br.readLine()) != null) {
                sb.append(lerroa).append("\n");
            }
        }
        return sb.toString();
    }

    /**
     * Testua garbitzen du: emailak, URLak, zenbakiak, puntuazioa kendu,
     * minuskula bihurtu eta zuriune anitzak ezabatu.
     *
     * @param testua garbitu beharreko testua
     * @return testu garbia
     */
    private String testuaGarbitu(String testua) {
    	testua = testua.replaceAll("\\S+@\\S+\\.\\S+", " ");
    	testua = testua.replaceAll("(http|https)://\\S+", " ");
    	testua = testua.replaceAll("\\b\\d+\\b", " ");
    	testua = testua.replaceAll("[^a-zA-Záéíóúñü' ]", " ");
    	testua = testua.toLowerCase();
        return testua.replaceAll("\\s+", " ").trim();
    }

    /**
     * Datak train eta test multzoetan banatzen ditu, klaseen proportzioa
     * mantenduz (hold‑out estratifikatua).
     *
     * @param guztiak   email guztiak
     * @param train entrenamendurako zerrenda
     * @param test  test‑erako zerrenda
     */
    private void dataZatitu(List<Email> guztiak, List<Email> train, List<Email> test) {
        List<Email> hamList = new ArrayList<>();
        List<Email> spamList = new ArrayList<>();
        for (Email e : guztiak) {
            if (e.mota.equals("ham")) hamList.add(e);
            else spamList.add(e);
        }
        Collections.shuffle(hamList, new Random(RANDOM_HAZIA));
        Collections.shuffle(spamList, new Random(RANDOM_HAZIA));
        
        int hamTrainKop = (int) Math.round(hamList.size() * TRAIN_RATIO);
        int spamTrainKop = (int) Math.round(spamList.size() * TRAIN_RATIO);
        
        train.addAll(hamList.subList(0, hamTrainKop));
        train.addAll(spamList.subList(0, spamTrainKop));
        test.addAll(hamList.subList(hamTrainKop, hamList.size()));
        test.addAll(spamList.subList(spamTrainKop, spamList.size()));
        
        Collections.shuffle(train, new Random(RANDOM_HAZIA));
        Collections.shuffle(test, new Random(RANDOM_HAZIA));
    }

    /**
     * Email zerrenda bat Wekako Instances objektu bihurtzen du.
     * <p>
     * Bi atributu sortzen ditu: "testua" (String) eta "class" (ham/spam).
     * </p>
     *
     * @param emails    email zerrenda
     * @param withClass true baldin badu klase atributua jartzen du; bestela dummy bat.
     * @return Instances objektua
     */
    private Instances instantziakSortu(List<Email> emails, boolean withClass) {
        ArrayList<Attribute> atributuak = new ArrayList<>();
        atributuak.add(new Attribute("testua", (ArrayList<String>) null));
        
        if (withClass) {
            ArrayList<String> klaseBalioak = new ArrayList<>();
            klaseBalioak.add("ham");
            klaseBalioak.add("spam");
            atributuak.add(new Attribute("klasea", klaseBalioak));
        } else {
            ArrayList<String> lista = new ArrayList<>();
            lista.add("?");
            atributuak.add(new Attribute("klasea", lista));
        }
        
        Instances data = new Instances("klasea", atributuak, emails.size());
        data.setClassIndex(data.numAttributes() - 1);

        for (Email e : emails) {
            double[] vals = new double[data.numAttributes()];
            vals[0] = data.attribute(0).addStringValue(e.testua);
            vals[1] = withClass ? data.attribute(1).indexOfValue(e.mota) : 0;
            data.add(new DenseInstance(1.0, vals));
        }
        return data;
    }
}
