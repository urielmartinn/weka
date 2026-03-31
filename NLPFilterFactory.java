package proiektua;

import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stopwords.Rainbow;
import weka.core.tokenizers.WordTokenizer;

/**
 * Clase utilitaria para centralizar la configuración del procesamiento de lenguaje natural (NLP).
 * Aplica el principio DRY (Don't Repeat Yourself).
 */
public class NLPFilterFactory {

    /**
     * Devuelve el filtro configurado con los parámetros empíricamente óptimos.
     * @param wordsToKeep El tamaño máximo del diccionario.
     * @return StringToWordVector configurado con TF-IDF, minúsculas, stemmer y stopwords.
     */
    public static StringToWordVector createSpamFilter(int wordsToKeep) {
        StringToWordVector filter = new StringToWordVector();
        filter.setLowerCaseTokens(true);
        filter.setWordsToKeep(wordsToKeep);
        filter.setStopwordsHandler(new Rainbow());
        filter.setStemmer(new LovinsStemmer());
        filter.setTokenizer(new WordTokenizer());
        
        // Configuración TF-IDF
        filter.setOutputWordCounts(true);
        filter.setTFTransform(true);
        filter.setIDFTransform(true);
        
        return filter;
    }
}
