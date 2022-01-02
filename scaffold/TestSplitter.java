import java.util.*;
import java.util.function.*;
import java.util.stream.*;

// Computes a random split for the given data if the target depth has not been reached.
public class TestSplitter implements Splitter {
    private double[][] matrix;
    private boolean[] labels;
    private Random random;
    private int depth;

    // The default value for the depth.
    private static final int DEFAULT_DEPTH = 2;

    // Constructs a new TestSplitter with the given design matrix and labels.
    public TestSplitter(double[][] matrix, boolean[] labels) {
        this(matrix, labels, new Random());
    }

    // Constructs a new TestSplitter with the given design matrix, labels, and randomness.
    public TestSplitter(double[][] matrix, boolean[] labels, Random random) {
        this(matrix, labels, random, DEFAULT_DEPTH);
    }

    // Constructs a new TestSplitter with the given design matrix, labels, randomness, and depth.
    public TestSplitter(double[][] matrix, boolean[] labels, Random random, int depth) {
        if (matrix.length != labels.length) {
            throw new IllegalArgumentException("matrix length != labels length");
        }
        this.matrix = matrix;
        this.labels = labels;
        this.random = random;
        this.depth = depth;
    }

    // Returns the best split and the left and right splitters, or null if no good split exists.
    public Result split() {
        if (depth == 0 || size() == 0) {
            return null;
        }
        int r = random.nextInt(size());
        int index = random.nextInt(matrix[r].length);
        double threshold = matrix[r][index];
        IntPredicate left = i -> matrix[i][index] <= threshold;
        IntPredicate right = left.negate();
        return new Splitter.Result(index, threshold, mask(left), mask(right));
    }

    // Returns a new TestSplitter containing only data where indices are true for the given predicate.
    private TestSplitter mask(IntPredicate predicate) {
        int[] indices = IntStream.range(0, size()).filter(predicate).toArray();
        double[][] newMatrix = new double[indices.length][];
        boolean[] newLabels = new boolean[indices.length];
        for (int i = 0; i < indices.length; i += 1) {
            newMatrix[i] = matrix[indices[i]];
            newLabels[i] = labels[indices[i]];
        }
        return new TestSplitter(newMatrix, newLabels, random, depth - 1);
    }

    // Returns the majority label for this splitter or false if the size() is 0.
    public boolean label() {
        int countTrue = 0;
        for (boolean label : labels) {
            if (label) {
                countTrue += 1;
            }
        }
        return countTrue > size() / 2;
    }

    // Returns the number of data points in this splitter.
    public int size() {
        return matrix.length;
    }
}
