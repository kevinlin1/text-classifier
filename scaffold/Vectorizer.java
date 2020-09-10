public interface Vectorizer {
    // Fits the model to the given texts and returns this instance.
    public Vectorizer fit(String... texts);

    // Fits the model to the given texts and returns the transformed design matrix. Equivalent to
    // fit followed by transform, but more efficiently implemented.
    public double[][] fitTransform(String... texts);

    // Returns the design matrix for the given texts.
    public double[][] transform(String... texts);
}
