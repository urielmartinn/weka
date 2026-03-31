package proiektua;

import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stopwords.Rainbow;
import weka.core.tokenizers.WordTokenizer;

/**
 * Prozesamendu linguistiko naturalerako (NLP) iragazkiak konfiguratzeko
 * klase utilitarioa. DRY (Don't Repeat Yourself) printzipioa aplikatzen du.
 * 
 */
public class IragazkiSortzailea {

    /**
     * Exekutagarri moduan erabiltzeko sarrera-puntua
     * 
     * Erabilera:
     * java proiektua.IragazkiSortzailea [wordsToKeep]
     */
    public static void main(String[] args) {
        int wordsToKeep = 1000;
        if (args.length >= 1) {
            try {
                wordsToKeep = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                System.err.println("wordsToKeep zenbakia izan behar da. Adib: 1000");
                System.exit(1);
            }
        }

        StringToWordVector iragazkia = sortuSpamIragazkia(wordsToKeep);

        String[] aukerak = iragazkia.getOptions();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < aukerak.length; i++) {
            if (i > 0) sb.append(' ');
            sb.append(aukerak[i]);
        }

        System.out.println("OK: StringToWordVector sortuta");
        System.out.println("Aukerak: " + sb);
    }

    /**
     * Iragazkia itzultzen du, esperimentuetan optimotzat jo diren
     * parametroekin konfiguratuta.
     * <p>
     * Konfigurazio honek TF‑IDF erabiltzen du, letra xeheak,
     * stopwords kentzea (Rainbow), Lovins stemmer-a eta tokenizatzaile
     * sinplea (hitzen araberakoa). WordsToKeep parametroak hiztegiaren
     * tamaina muga ezartzen du.
     * </p>
     *
     * @param wordsToKeep Hiztegian mantenduko diren hitz kopuru maximoa
     * @return StringToWordVector iragazkia, spam detekziorako prestatuta
     */
    public static StringToWordVector sortuSpamIragazkia(int wordsToKeep) {
        StringToWordVector iragazkia = new StringToWordVector();
        iragazkia.setLowerCaseTokens(true);
        iragazkia.setWordsToKeep(wordsToKeep);
        iragazkia.setStopwordsHandler(new Rainbow());
        iragazkia.setStemmer(new LovinsStemmer());
        iragazkia.setTokenizer(new WordTokenizer());
        
        // TF‑IDF konfigurazioa
        iragazkia.setOutputWordCounts(true);
        iragazkia.setTFTransform(true);
        iragazkia.setIDFTransform(true);
        
        return iragazkia;
    }
}
