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
