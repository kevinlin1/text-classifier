import java.io.*;
import java.util.*;

public class TextClassifier {
    // TODO: Your code here

    // An internal node or a leaf node in the decision tree.
    private static class Node {
        public final int index;
        public final double threshold;
        public final boolean label;
        public Node left;
        public Node right;

        // Constructs a new leaf node with the given label.
        public Node(boolean label) {
            this(0, 0.0, label, null, null);
        }

        // Constructs a new node with the given index, threshold, label, and left and right nodes.
        public Node(int index, double threshold, boolean label, Node left, Node right) {
            this.index = index;
            this.threshold = threshold;
            this.label = label;
            this.left = left;
            this.right = right;
        }

        // Returns true if and only if this node is a leaf node.
        public boolean isLeaf() {
            return left == null && right == null;
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            throw new IllegalArgumentException("java TextClassifier [tsv file]");
        }
        String filename = args[0];
        int N = 5000;
        boolean[] labels = new boolean[N];
        String[] corpus = new String[N];

        Scanner input = new Scanner(new File(filename));
        input.nextLine(); // Skip header
        for (int i = 0; i < N; i += 1) {
            Scanner line = new Scanner(input.nextLine()).useDelimiter("\t");
            labels[i] = line.nextBoolean();
            corpus[i] = line.next();
        }

        Vectorizer vectorizer = new Vectorizer();
        Splitter splitter = new GiniSplitter(vectorizer.fitTransform(corpus), labels);
        TextClassifier clf = new TextClassifier(vectorizer, splitter);
        clf.print();

        int correct = 0;
        for (int i = 0; i < N; i += 1) {
            if (clf.classify(corpus[i]) == labels[i]) {
                correct += 1;
            }
        }
        System.out.println("Training accuracy: " + correct / (double) N);
    }
}
