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

    public static void main(String[] args) throws FileNotFoundException {
        if (args.length != 1) {
            throw new IllegalArgumentException("java TextClassifier [tsv file]");
        }
        File file = new File(args[0]);
        Scanner input = new Scanner(file);
        int N = 0;
        input.nextLine(); // Skip header
        while (input.hasNextLine()) {
            N += 1;
            input.nextLine();
        }
        boolean[] labels = new boolean[N];
        String[] messages = new String[N];
        input = new Scanner(file);
        input.nextLine(); // Skip header
        for (int i = 0; i < N; i += 1) {
            Scanner line = new Scanner(input.nextLine()).useDelimiter("\t");
            labels[i] = line.nextBoolean();
            messages[i] = line.next();
        }

        Vectorizer vectorizer = new Vectorizer();
        Splitter splitter = new GiniSplitter(vectorizer.fitTransform(messages), labels);
        TextClassifier clf = new TextClassifier(vectorizer, splitter);
        clf.prune(10);
        clf.print();
    }
}
