import smile.math.matrix.*;

// Latent semantic analysis text vectorizer.
public class LSAVectorizer implements Vectorizer {
    // Basic text vectorizer that will be simplified through latent semantic analysis.
    private BM25Vectorizer vectorizer;
    // Truncated term matrix for projecting from the BM25 vector space to the LSA vector space.
    private Matrix components;

    // Number of components to retain in the final design matrix.
    private static final int K = 50;

    // Constructs an empty LSAVectorizer model with the default preprocessor.
    public LSAVectorizer() {
        this(Preprocessor.DEFAULT);
    }

    // Constructs an empty LSAVectorizer model with the given preprocessor.
    public LSAVectorizer(Preprocessor preprocessor) {
        this.vectorizer = new BM25Vectorizer(preprocessor);
        this.components = null;
    }

    // Fits the model to the given texts and returns this instance.
    public Vectorizer fit(String... texts) {
        fitMatrix(true, texts);
        return this;
    }

    // Fits the model to the given texts and returns the BM25+ design matrix for the texts.
    private Matrix fitMatrix(boolean overwrite, String... texts) {
        // Fit the BM25+ vectorizer and compute the design matrix for the corpus.
        Matrix X = new Matrix(vectorizer.fitTransform(texts));
        // Save the truncated term matrix from singular value decomposition.
        // https://scikit-learn.org/0.23/modules/decomposition.html#lsa
        Matrix V = X.svd(true, overwrite).V;
        components = V.submatrix(0, 0, V.nrows() - 1, Math.min(K, V.ncols()) - 1).clone();
        return X;
    }

    // Fits the model to the given texts and returns the transformed design matrix. Equivalent to
    // fit followed by transform, but more efficiently implemented.
    public double[][] fitTransform(String... texts) {
        Matrix X = fitMatrix(false, texts);
        return X.mm(components).toArray();
    }

    // Returns the design matrix for the given texts.
    public double[][] transform(String... texts) {
        if (components == null) {
            throw new IllegalStateException("must fit before transform");
        }
        Matrix X = new Matrix(vectorizer.transform(texts));
        return X.mm(components).toArray();
    }
}
