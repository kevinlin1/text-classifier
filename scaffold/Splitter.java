import java.util.*;
import java.util.function.*;
import java.util.stream.*;

// Computes the best split for the given data based on Gini impurity and information gain.
public class Splitter {
    private double[][] matrix;
    private boolean[] labels;
    private int originalSize;
    private int countTrue;
    private double impurity;

    // The minimum impurity improvement required to continue splitting.
    private static final double MIN_IMPURITY_DECREASE = 0.01;
    // The minimum number of data points required to continue splitting.
    private static final int MIN_SIZE_SPLIT = 5;

    // Constructs a new Splitter with the given design matrix and binary labels.
    public Splitter(double[][] matrix, boolean[] labels) {
        this(matrix, labels, matrix.length);
    }

    // Constructs a new Splitter with the given design matrix, binary labels, and original size.
    private Splitter(double[][] matrix, boolean[] labels, int originalSize) {
        if (matrix.length != labels.length) {
            throw new IllegalArgumentException("matrix length != labels length");
        }
        this.matrix = matrix;
        this.labels = labels;
        this.originalSize = originalSize;
        this.countTrue = 0;
        for (boolean label : labels) {
            if (label) {
                this.countTrue += 1;
            }
        }
        this.impurity = impurity(this.countTrue);
    }

    // The left and right splitters that result from applying a split.
    public static class Result {
        public final Split split;
        public final Splitter left;
        public final Splitter right;

        private Result(Split split, Splitter left, Splitter right) {
            this.split = split;
            this.left = left;
            this.right = right;
        }
    }

    // Returns the best split and the left and right splitters, or null if no good split exists.
    public Result split() {
        if (size() < MIN_SIZE_SPLIT) {
            return null;
        }
        BestSplit bestSplit = (
            Arrays.stream(featureSplits())
                  .parallel()
                  .map(FeatureSplit::best)
                  .filter(BestSplit::isPresent)
                  .filter(b -> (size() / (double) originalSize) * b.gain >= MIN_IMPURITY_DECREASE)
                  .max(BestSplit::compareTo)
                  .orElseGet(BestSplit::empty)
        );
        if (!bestSplit.isPresent()) {
            return null;
        }
        IntPredicate left = i -> bestSplit.split.goLeft(matrix[i]);
        return new Result(bestSplit.split, mask(left), mask(left.negate()));
    }

    // Returns the majority label for this splitter.
    public boolean label() {
        return countTrue > size() / 2;
    }

    // Returns the number of data points in this splitter.
    public int size() {
        return matrix.length;
    }

    // Returns the Gini impurity given the count of either class in binary classification.
    private double impurity(int count) {
        if (count == 0 || count == size()) {
            return 0.0;
        }
        double p = count / (double) size();
        return 1 - ((p * p) + ((1 - p) * (1 - p)));
    }

    // Returns a new Splitter containing only data where indices are true for the given predicate.
    private Splitter mask(IntPredicate predicate) {
        int[] indices = IntStream.range(0, matrix.length).filter(predicate).toArray();
        double[][] newMatrix = new double[indices.length][];
        boolean[] newLabels = new boolean[indices.length];
        for (int i = 0; i < indices.length; i += 1) {
            newMatrix[i] = matrix[indices[i]];
            newLabels[i] = labels[indices[i]];
        }
        return new Splitter(newMatrix, newLabels, originalSize);
    }

    // Returns an array of all possible feature splits for each matrix variable.
    private FeatureSplit[] featureSplits() {
        int m = matrix.length;
        int n = matrix[0].length;
        FeatureSplit[] result = new FeatureSplit[n];
        for (int j = 0; j < n; j += 1) {
            SortedSet<Split> splits = new TreeSet<>();
            for (int i = 0; i < m; i += 1) {
                splits.add(new Split(j, matrix[i][j]));
            }
            result[j] = new FeatureSplit(j, splits);
        }
        return result;
    }

    // All splits for a particular column (feature) in the design matrix.
    private class FeatureSplit {
        public int column;
        public SortedSet<Split> splits;

        // Constructs a new FeatureSplit with the given column and set of splits.
        private FeatureSplit(int column, SortedSet<Split> splits) {
            this.column = column;
            this.splits = splits;
        }

        // Returns the best split in this feature.
        public BestSplit best() {
            BestSplit best = BestSplit.empty();
            for (Split proposed : splits) {
                int countCorrectSplit = 0;
                for (int i = 0; i < size(); i += 1) {
                    if (labels[i] && proposed.goLeft(matrix[i])) {
                        countCorrectSplit += 1;
                    }
                }
                double gain = impurity - splitImpurity(countCorrectSplit);
                if (gain > best.gain) {
                    best.update(proposed, gain);
                }
            }
            return best;
        }

        // Returns the weighted impurity for the proposed split given the count of either class.
        private double splitImpurity(int count) {
            int other = size() - count;
            return (count * impurity(count) + other * impurity(other)) / (double) size();
        }
    }

    // Two-element, mutable container representing the best split and its gain.
    private static class BestSplit implements Comparable<BestSplit> {
        public Split split;
        public double gain;

        // Constructs a new BestSplit container with the given split and gain.
        public BestSplit(Split split, double gain) {
            this.split = split;
            this.gain = gain;
        }

        // Returns a new empty split.
        public static BestSplit empty() {
            return new BestSplit(null, 0.0);
        }

        // Returns true if the best split is not empty.
        public boolean isPresent() {
            return split != null;
        }

        // Updates the split and gain stored in this container.
        public void update(Split newSplit, double newGain) {
            split = newSplit;
            gain = newGain;
        }

        // Compares to the other best split by gain value.
        public int compareTo(BestSplit other) {
            return Double.compare(this.gain, other.gain);
        }
    }
}
