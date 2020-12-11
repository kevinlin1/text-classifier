import java.util.*;
import java.util.function.*;
import java.util.stream.*;

// Computes the best split for the given data based on Gini impurity and information gain.
public class GiniSplitter implements Splitter {
    private double[][] matrix;
    private boolean[] labels;
    private int originalSize;
    private double impurity;
    private boolean label;

    // The minimum impurity improvement required to continue splitting.
    private static final double MIN_IMPURITY_DECREASE = 0.001;
    // The minimum number of data points required to continue splitting.
    private static final int MIN_SIZE_SPLIT = 5;

    // Constructs a new GiniSplitter with the given design matrix and labels.
    public GiniSplitter(double[][] matrix, boolean[] labels) {
        this(matrix, labels, matrix.length);
    }

    // Constructs a new GiniSplitter with the given design matrix, labels, and original size.
    private GiniSplitter(double[][] matrix, boolean[] labels, int originalSize) {
        if (matrix.length != labels.length) {
            throw new IllegalArgumentException("matrix length != labels length");
        }
        this.matrix = matrix;
        this.labels = labels;
        this.originalSize = originalSize;
        int countTrue = 0;
        for (boolean label : labels) {
            if (label) {
                countTrue += 1;
            }
        }
        this.impurity = impurity(countTrue);
        this.label = countTrue > size() / 2;
    }

    // Returns the optimal Splitter.Result representing the split with the maximum information gain
    // or null if no valid split exists.
    public Splitter.Result split() {
        if (size() < MIN_SIZE_SPLIT) {
            return null;
        }
        double subsample = size() / (double) originalSize;
        Split max = (
            IntStream.range(0, matrix[0].length)
                     .parallel()
                     .mapToObj(this::split)
                     .max(Comparator.comparingDouble(s -> s.gain))
                     .filter(s -> subsample * s.gain >= MIN_IMPURITY_DECREASE)
                     .orElse(null)
        );
        if (max == null) {
            return null;
        }
        IntPredicate left = i -> matrix[i][max.index] <= max.threshold;
        IntPredicate right = left.negate();
        return new Splitter.Result(max.index, max.threshold, mask(left), mask(right));
    }

    // Returns the split with the maximum information gain for the given index (feature).
    private Split split(int index) {
        SortedSet<Double> thresholds = new TreeSet<>();
        for (int i = 0; i < size(); i += 1) {
            thresholds.add(matrix[i][index]);
        }
        double bestThreshold = Double.NaN;
        double bestGain = 0.0;
        for (double threshold : thresholds) {
            double gain = informationGain(index, threshold);
            if (gain > bestGain) {
                bestThreshold = threshold;
                bestGain = gain;
            }
        }
        return new Split(index, bestThreshold, bestGain);
    }

    // Immutable container representing a possible split.
    private static class Split {
        public final int index;
        public final double threshold;
        public final double gain;

        // Constructs a new Split with the given index, threshold, and gain.
        public Split(int index, double threshold, double gain) {
            this.index = index;
            this.threshold = threshold;
            this.gain = gain;
        }
    }

    // Returns the Gini impurity given the count of either class in binary classification.
    private double impurity(int count) {
        if (count == 0 || count == size()) {
            return 0.0;
        }
        double p = count / (double) size();
        return 1 - ((p * p) + ((1 - p) * (1 - p)));
    }

    // Returns the information gain for applying a split with the given index and threshold.
    private double informationGain(int index, double threshold) {
        int correct = 0;
        for (int i = 0; i < size(); i += 1) {
            if (labels[i] && matrix[i][index] <= threshold) {
                correct += 1;
            }
        }
        int incorrect = size() - correct;
        double weightedSplit = correct * impurity(correct) + incorrect * impurity(incorrect);
        return this.impurity - weightedSplit / size();
    }

    // Returns a new GiniSplitter containing only data where indices are true for the given predicate.
    private GiniSplitter mask(IntPredicate predicate) {
        int[] indices = IntStream.range(0, size()).filter(predicate).toArray();
        double[][] newMatrix = new double[indices.length][];
        boolean[] newLabels = new boolean[indices.length];
        for (int i = 0; i < indices.length; i += 1) {
            newMatrix[i] = matrix[indices[i]];
            newLabels[i] = labels[indices[i]];
        }
        return new GiniSplitter(newMatrix, newLabels, originalSize);
    }

    // Returns the majority label for this splitter.
    public boolean label() {
        return label;
    }

    // Returns the number of data points in this splitter.
    public int size() {
        return matrix.length;
    }
}
