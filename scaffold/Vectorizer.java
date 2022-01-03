import java.util.*;
import java.util.regex.*;
import java.util.stream.*;

// Okapi BM25+ term-importance text vectorizer.
public class Vectorizer {
    // The average number of terms per document in the corpus.
    private double averageLength;
    // Feature representation mapping integer to string word.
    private String[] features;
    // Inverse document frequency for each feature to reduce the importance of frequent terms.
    private double[] idf;

    // Maximum proportion of documents that a term can appear in.
    private static final double MAX_DF = 0.05;
    // Minimum proportion of documents that a term needs to appear.
    private static final double MIN_DF = 0.002;
    // BM25 calibration parameter for term-frequency scaling.
    private static final double K1 = 1.2;
    // BM25 calibration parameter for document length scaling.
    private static final double B = 0.75;
    // BM25 calibration parameter for long, matching documents.
    private static final double DELTA = 1.0;

    // Constructs an unfitted Vectorizer model.
    public Vectorizer() {
        this.averageLength = 0.0;
        this.features = null;
        this.idf = null;
    }

    // Fits the model to the given texts and returns this instance.
    public Vectorizer fit(String... texts) {
        fitStream(texts);
        return this;
    }

    // Fits the model to the given texts and returns a stream of processed texts.
    private Stream<BagOfWords> fitStream(String... texts) {
        int N = texts.length;
        BagOfWords[] corpus = new BagOfWords[N];
        Map<String, Integer> df = new HashMap<>();
        averageLength = 0.0;
        for (int i = 0; i < N; i += 1) {
            corpus[i] = BagOfWords.from(texts[i]);
            for (String word : corpus[i].unique()) {
                df.put(word, df.getOrDefault(word, 0) + 1);
            }
            averageLength += corpus[i].size();
        }
        averageLength /= N;
        features = df.keySet().stream().filter(
            word -> (MIN_DF * N) <= df.get(word) && df.get(word) <= (MAX_DF * N)
        ).toArray(String[]::new);
        idf = Arrays.stream(features).mapToDouble(
            word -> Math.log((N - df.get(word) + 0.5) / (df.get(word) + 0.5))
        ).toArray();
        return Arrays.stream(corpus);
    }

    // Fits the model to the given texts and returns the transformed design matrix. Equivalent to
    // fit followed by transform, but more efficiently implemented.
    public double[][] fitTransform(String... texts) {
        return matrix(fitStream(texts));
    }

    // Returns the name of the feature for the given index.
    public String getFeature(int index) {
        if (averageLength == 0.0 || features == null || idf == null) {
            throw new IllegalStateException("must fit before getFeature");
        }
        return features[index];
    }

    // Returns the design matrix for the given texts.
    public double[][] transform(String... texts) {
        if (averageLength == 0.0 || features == null || idf == null) {
            throw new IllegalStateException("must fit before transform");
        }
        return matrix(Arrays.stream(texts).map(BagOfWords::from));
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

    // Bag-of-words text representation that stores term-frequency counts.
    private static class BagOfWords {
        private Map<String, Integer> counts;
        private int size;

        // Returns a new BagOfWords with the given words.
        private BagOfWords(Stream<String> words) {
            this.counts = new HashMap<>();
            this.size = 0;
            words.forEach(word -> {
                counts.merge(word.toLowerCase(), 1, Integer::sum);
                size += 1;
            });
        }

        // Returns a new BagOfWords after tokenizing and stemming the given text.
        public static BagOfWords from(String text) {
            return new BagOfWords(Tokenizer.tokenize(text).map(Stemmer::stem));
        }

        // Returns the total number of words in this bag.
        public int size() {
            return size;
        }

        // Returns the unique words in this bag.
        public Set<String> unique() {
            return counts.keySet();
        }

        // Returns the term-frequency count for the given term.
        public int tf(String term) {
            return counts.getOrDefault(term, 0);
        }
    }

    // Tokenizes a string using algorithms by Grefenstette (1999) and Palmer (2000).
    private static class Tokenizer {
        // A regular expression for letters and numbers.
        private static final String RE_LETTER_NUMBER = "[a-zA-Z0-9]";
        // A regular expression for non-letters and non-numbers.
        private static final String RE_NOT_LETTER_NUMBER = "[^a-zA-Z0-9]";
        // A regular expression for separators.
        private static final String RE_SEPARATOR = "[\\?!()\";/\\|`]";
        // A regular expression for separators.
        private static final String RE_CLITICS =
                "'|:|-|'S|'D|'M|'LL|'RE|'VE|N'T|'s|'d|'m|'ll|'re|'ve|n't";

        public static final Stream<String> tokenize(String text) {
            // Change tabs to spaces.
            text = text.replaceAll("\\t", " ");
            // Put blanks around unambiguous separators.
            text = text.replaceAll("(" + RE_SEPARATOR + ")", " $1 ");
            // Put blanks around commas.
            text = text.replaceAll("([^\\s]),", "$1 ,");
            text = text.replaceAll(",([^\\s])", " , $1");
            // Distinguish single quotes from apstrophes.
            text = text.replaceAll("^(')", "$1 ");
            text = text.replaceAll("(" + RE_NOT_LETTER_NUMBER + ")'", "$1 '");
            // Segment unambiguous word-final clitics and punctuations.
            text = text.replaceAll("(" + RE_CLITICS + ")$", " $1");
            text = text.replaceAll("(" + RE_CLITICS + ")(" + RE_NOT_LETTER_NUMBER + ")", " $1 $2");
            // Deal with periods.
            String[] words = text.trim().split("\\s+");
            Pattern p1 = Pattern.compile(".*" + RE_LETTER_NUMBER + "\\.");
            Pattern p2 = Pattern.compile("^([A-Za-z]\\.([A-Za-z]\\.)+|[A-Z][bcdfghj-nptvxz]+\\.)$");
            Stream.Builder<String> tokens = Stream.builder();
            for (String word : words) {
                Matcher m1 = p1.matcher(word);
                Matcher m2 = p2.matcher(word);
                tokens.add(word);
            }
            return tokens.build();
        }
    }

    // Porter stemming algorithm to simplify words: https://tartarus.org/martin/PorterStemmer/
    private static class Stemmer {
        private static char[] b;
        private static int j;
        private static int k;

        public static String stem(String word) {
            b = word.toCharArray();
            k = word.length() - 1;
            if (k > 1) {
                step1();
                step2();
                step3();
                step4();
                step5();
                step6();
            }
            return new String(b, 0, k + 1);
        }

        private static boolean cons(int i) {
            switch (b[i]) {
                case 'a':
                case 'e':
                case 'i':
                case 'o':
                case 'u':
                    return false;
                case 'y':
                    return (i == 0) ? true : !cons(i - 1);
                default:
                    return true;
            }
        }

        private static int m() {
            int n = 0;
            int i = 0;
            while (true) {
                if (i > j) {
                    return n;
                }
                if (!cons(i)) {
                    break;
                }
                i++;
            }
            i++;
            while (true) {
                while (true) {
                    if (i > j) {
                        return n;
                    }
                    if (cons(i)) {
                        break;
                    }
                    i++;
                }
                i++;
                n++;
                while (true) {
                    if (i > j) {
                        return n;
                    }
                    if (!cons(i)) {
                        break;
                    }
                    i++;
                }
                i++;
            }
        }

        private static boolean vowels() {
            int i;
            for (i = 0; i <= j; i++) {
                if (!cons(i)) {
                    return true;
                }
            }
            return false;
        }

        private static boolean doublec(int j) {
            if (j < 1) {
                return false;
            }
            if (b[j] != b[j - 1]) {
                return false;
            }
            return cons(j);
        }

        private static boolean cvc(int i) {
            if (i < 2 || !cons(i) || cons(i - 1) || !cons(i - 2)) {
                return false;
            }
            int ch = b[i];
            if (ch == 'w' || ch == 'x' || ch == 'y') {
                return false;
            }
            return true;
        }

        private static boolean ends(String s) {
            int l = s.length();
            int o = k - l + 1;
            if (o < 0) {
                return false;
            }
            for (int i = 0; i < l; i++) {
                if (b[o + i] != s.charAt(i)) {
                    return false;
                }
            }
            j = k - l;
            return true;
        }

        private static void set(String s) {
            int l = s.length();
            int o = j + 1;
            for (int i = 0; i < l; i++) {
                b[o + i] = s.charAt(i);
            }
            k = j + l;
        }

        private static void r(String s) {
            if (m() > 0) {
                set(s);
            }
        }

        private static void step1() {
            step1(false);
        }

        private static void step1(boolean y) {
            if (b[k] == 's') {
                if (ends("sses")) {
                    k -= 2;
                } else if (ends("ies")) {
                    if (y && k - 3 >= 0 && cons(k - 3)) {
                        set("y");
                    } else {
                        set("i");
                    }
                } else if (b[k - 1] != 's') {
                    k--;
                }
            }
            if (ends("eed")) {
                if (m() > 0) {
                    k--;
                }
            } else if ((ends("ed") || ends("ing")) && vowels()) {
                k = j;
                if (ends("at")) {
                    set("ate");
                } else if (ends("bl")) {
                    set("ble");
                } else if (ends("iz")) {
                    set("ize");
                } else if (y && ends("i") && k - 1 >= 0 && cons(k - 1)) {
                    set("y");
                } else if (doublec(k)) {
                    k--;
                    int ch = b[k];
                    if (ch == 'l' || ch == 's' || ch == 'z') {
                        k++;
                    }
                } else if (m() == 1 && cvc(k)) {
                    set("e");
                }
            }
        }

        private static void step2() {
            if (ends("y") && vowels()) {
                b[k] = 'i';
            }
        }

        private static void step3() {
            if (k == 0) {
                return;
            }
            switch (b[k - 1]) {
                case 'a':
                    if (ends("ational")) {
                        r("ate");
                        break;
                    }
                    if (ends("tional")) {
                        r("tion");
                        break;
                    }
                    break;
                case 'c':
                    if (ends("enci")) {
                        r("ence");
                        break;
                    }
                    if (ends("anci")) {
                        r("ance");
                        break;
                    }
                    break;
                case 'e':
                    if (ends("izer")) {
                        r("ize");
                        break;
                    }
                    break;
                case 'l':
                    if (ends("bli")) {
                        r("ble");
                        break;
                    }
                    if (ends("alli")) {
                        r("al");
                        break;
                    }
                    if (ends("entli")) {
                        r("ent");
                        break;
                    }
                    if (ends("eli")) {
                        r("e");
                        break;
                    }
                    if (ends("ousli")) {
                        r("ous");
                        break;
                    }
                    break;
                case 'o':
                    if (ends("ization")) {
                        r("ize");
                        break;
                    }
                    if (ends("ation")) {
                        r("ate");
                        break;
                    }
                    if (ends("ator")) {
                        r("ate");
                        break;
                    }
                    break;
                case 's':
                    if (ends("alism")) {
                        r("al");
                        break;
                    }
                    if (ends("iveness")) {
                        r("ive");
                        break;
                    }
                    if (ends("fulness")) {
                        r("ful");
                        break;
                    }
                    if (ends("ousness")) {
                        r("ous");
                        break;
                    }
                    break;
                case 't':
                    if (ends("aliti")) {
                        r("al");
                        break;
                    }
                    if (ends("iviti")) {
                        r("ive");
                        break;
                    }
                    if (ends("biliti")) {
                        r("ble");
                        break;
                    }
                    break;
                case 'g':
                    if (ends("logi")) {
                        r("log");
                        break;
                    }
            }
        }

        private static void step4() {
            switch (b[k]) {
                case 'e':
                    if (ends("icate")) {
                        r("ic");
                        break;
                    }
                    if (ends("ative")) {
                        r("");
                        break;
                    }
                    if (ends("alize")) {
                        r("al");
                        break;
                    }
                    break;
                case 'i':
                    if (ends("iciti")) {
                        r("ic");
                        break;
                    }
                    break;
                case 'l':
                    if (ends("ical")) {
                        r("ic");
                        break;
                    }
                    if (ends("ful")) {
                        r("");
                        break;
                    }
                    break;
                case 's':
                    if (ends("ness")) {
                        r("");
                        break;
                    }
                    break;
            }
        }

        private static void step5() {
            if (k == 0) {
                return;
            }
            switch (b[k - 1]) {
                case 'a':
                    if (ends("al")) {
                        break;
                    }
                    return;
                case 'c':
                    if (ends("ance")) {
                        break;
                    }
                    if (ends("ence")) {
                        break;
                    }
                    return;
                case 'e':
                    if (ends("er")) {
                        break;
                    }
                    return;
                case 'i':
                    if (ends("ic")) {
                        break;
                    }
                    return;
                case 'l':
                    if (ends("able")) {
                        break;
                    }
                    if (ends("ible")) {
                        break;
                    }
                    return;
                case 'n':
                    if (ends("ant")) {
                        break;
                    }
                    if (ends("ement")) {
                        break;
                    }
                    if (ends("ment")) {
                        break;
                    }
                    if (ends("ent")) {
                        break;
                    }
                    return;
                case 'o':
                    if (ends("ion") && j >= 0 && (b[j] == 's' || b[j] == 't')) {
                        break;
                    }
                    if (ends("ou")) {
                        break;
                    }
                    return;
                case 's':
                    if (ends("ism")) {
                        break;
                    }
                    return;
                case 't':
                    if (ends("ate")) {
                        break;
                    }
                    if (ends("iti")) {
                        break;
                    }
                    return;
                case 'u':
                    if (ends("ous")) {
                        break;
                    }
                    return;
                case 'v':
                    if (ends("ive")) {
                        break;
                    }
                    return;
                case 'z':
                    if (ends("ize")) {
                        break;
                    }
                    return;
                default:
                    return;
            }
            if (m() > 1) {
                k = j;
            }
        }

        private static void step6() {
            j = k;
            if (b[k] == 'e') {
                int a = m();
                if (a > 1 || a == 1 && !cvc(k - 1)) {
                    k--;
                }
            }
            if (b[k] == 'l' && doublec(k) && m() > 1) {
                k--;
            }
        }
    }
}
