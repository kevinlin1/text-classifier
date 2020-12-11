// Splitters provide a split method for dividing the given data points into left and right.
public interface Splitter {
    // Returns the best split and the left and right splitters, or null if no good split exists.
    public Result split();

    // Returns the majority label for this splitter.
    public boolean label();

    // Returns the number of data points in this splitter.
    public int size();

    // The index and threshold representing a split point, and the left and right splitters that
    // result from applying the split.
    public static class Result {
        public final int index;
        public final double threshold;
        public final Splitter left;
        public final Splitter right;

        protected Result(int index, double threshold, Splitter left, Splitter right) {
            this.index = index;
            this.threshold =  threshold;
            this.left = left;
            this.right = right;
        }
    }
}
