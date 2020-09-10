import java.util.*;

// A decision rule for splitting vector data.
public class Split implements Comparable<Split> {
    private final int index;
    private final double threshold;

    // Constructs a new Split with the given index into the vector and threshold value.
    public Split(int index, double threshold) {
        this.index = index;
        this.threshold = threshold;
    }

    // Returns true if and only if the value lies to the left of this decision rule.
    public boolean goLeft(double value) {
        return value <= threshold;
    }

    // Returns true if and only if the vector lies to the left of this decision rule.
    public boolean goLeft(double[] vector) {
        return goLeft(vector[index]);
    }

    // Returns a string representation of this decision rule.
    public String toString() {
        return "vector[" + index + "] <= " + threshold;
    }

    // Returns true if and only if o is a Split representing the same decision rule.
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        } else if (!(o instanceof Split)) {
            return false;
        }
        Split other = (Split) o;
        return this.index == other.index && Double.compare(this.threshold, other.threshold) == 0;
    }

    // Compares to the other Split by index and then threshold value.
    public int compareTo(Split other) {
        int cmp = Integer.compare(this.index, other.index);
        if (cmp == 0) {
            cmp = Double.compare(this.threshold, other.threshold);
        }
        return cmp;
    }

    // Returns a hash code for this decision rule.
    public int hashCode() {
        return Objects.hash(index, threshold);
    }
}
