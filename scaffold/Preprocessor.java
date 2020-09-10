import java.util.*;
import java.util.stream.*;

import smile.nlp.tokenizer.*;
import smile.nlp.normalizer.*;
import smile.nlp.dictionary.*;
import smile.nlp.stemmer.*;

// Text preprocessing pipeline via the Smile NLP examples: https://haifengl.github.io/nlp.html
public class Preprocessor {
    public static final Preprocessor DEFAULT = new Preprocessor(
        SimpleNormalizer.getInstance(),
        new SimpleTokenizer(true),
        new PorterStemmer()
    );

    private Normalizer normalizer;
    private Tokenizer tokenizer;
    private Stemmer stemmer;

    public Preprocessor(Normalizer normalizer, Tokenizer tokenizer, Stemmer stemmer) {
        this.normalizer = normalizer;
        this.tokenizer = tokenizer;
        this.stemmer = stemmer;
    }

    public BagOfWords process(String text) {
        return processWords(Arrays.stream(tokenizer.split(normalizer.normalize(text))));
    }

    public BagOfWords processWords(Stream<String> words) {
        return new BagOfWords(
                words.map(String::toLowerCase)
                     .filter(w -> !(
                         EnglishStopWords.DEFAULT.contains(w)
                         || EnglishPunctuations.getInstance().contains(w)
                     ))
                     .map(stemmer::stem)
                     .collect(Collectors.toList())
        );
    }
}
