import java.util.*;
import java.util.function.*;
import java.util.stream.*;

// Okapi BM25+ term-importance text vectorizer.
public class BM25Vectorizer implements Vectorizer {
    // The average number of terms per document in the corpus.
    private double averageLength;
    // Mapping integer to string word.
    private String[] features;
    // Inverse document frequency for each feature.
    private double[] idf;
    // String-to-BagOfWords preprocessor.
    private Preprocessor preprocessor;

    // Minimum number of documents that a term needs to appear.
    private static final int MIN_DF = 20;
    // BM25 calibration parameter for term-frequency scaling.
    private static final double K1 = 1.2;
    // BM25 calibration parameter for document length scaling.
    private static final double B = 0.75;
    // BM25 calibration parameter for long, matching documents.
    private static final double DELTA = 1.0;

    // Constructs an empty BM25Vectorizer model with the default preprocessor.
    public BM25Vectorizer() {
        this(Preprocessor.DEFAULT);
    }

    // Constructs an empty BM25Vectorizer model with the given preprocessor.
    public BM25Vectorizer(Preprocessor preprocessor) {
        this.averageLength = 0.0;
        this.features = null;
        this.idf = null;
        this.preprocessor = preprocessor;
    }

    // Fits the model to the given texts and returns this instance.
    public Vectorizer fit(String... texts) {
        fitStream(texts);
        return this;
    }

    // Fits the model to the given texts and returns a stream of processed texts.
    private Stream<BagOfWords> fitStream(String... texts) {
        // Compute the bag-of-words Document model (word count) for each text.
        BagOfWords[] corpus = new BagOfWords[texts.length];
        Map<String, Integer> df = new HashMap<>();
        averageLength = 0.0;
        for (int i = 0; i < texts.length; i += 1) {
            corpus[i] = preprocessor.process(texts[i]);
            for (String word : corpus[i].unique()) {
                df.put(word, df.getOrDefault(word, 0) + 1);
            }
            averageLength += corpus[i].size();
        }
        averageLength /= corpus.length;
        // Extract a vector feature representation based on words in the documents.
        features = df.keySet().stream().filter(w -> df.get(w) >= MIN_DF).toArray(String[]::new);
        // Compute the inverse document frequency to reduce the importance of common terms.
        idf = Arrays.stream(features).mapToDouble(
            word -> Math.log((corpus.length - df.get(word) + 0.5) / (df.get(word) + 0.5))
        ).toArray();
        return Arrays.stream(corpus);
    }

    // Fits the model to the given texts and returns the transformed design matrix. Equivalent to
    // fit followed by transform, but more efficiently implemented.
    public double[][] fitTransform(String... texts) {
        return matrix(fitStream(texts));
    }

    // Returns the design matrix for the given texts.
    public double[][] transform(String... texts) {
        if (averageLength == 0 || features == null || idf == null) {
            throw new IllegalStateException("must fit before transform");
        }
        return matrix(Arrays.stream(texts).map(preprocessor::process));
    }

    // Returns the design matrix for the BM25+ representation of the given documents.
    private double[][] matrix(Stream<BagOfWords> documents) {
        return documents.parallel().map(this::vector).toArray(double[][]::new);
    }

    // Returns the design vector for the BM25+ representation of the given document.
    private double[] vector(BagOfWords document) {
        return IntStream.range(0, features.length).mapToDouble(
            j -> idf[j] * tfn(document.tf(features[j]), document.size() / averageLength)
        ).toArray();
    }

    // Returns the BM25+ normalized term frequency value.
    // http://sifaka.cs.uiuc.edu/~ylv2/pub/cikm11-lowerbound.pdf
    private static double tfn(double tf, double n) {
        return ((tf * (K1 + 1)) / (tf + K1 * ((1 - B) + B * n))) + DELTA;
    }
}
