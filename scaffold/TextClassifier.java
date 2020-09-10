public class TextClassifier {
    // TODO: Your code here

    // Returns a new TextClassifier fitted against the given corpus and labels.
    public static TextClassifier from(String[] corpus, boolean[] labels) {
        Vectorizer vectorizer = new BM25Vectorizer();
        Splitter splitter = new Splitter(vectorizer.fitTransform(corpus), labels);
        return new TextClassifier(vectorizer, splitter);
    }

    // An internal node or a leaf node in the decision tree.
    private static class Node {
        public Split split;
        public boolean label;
        public Node left;
        public Node right;

        // Constructs a new leaf node with the given label.
        public Node(boolean label) {
            this(null, label, null, null);
        }

        // Constructs a new internal node with the given split, label, and left and right nodes.
        public Node(Split split, boolean label, Node left, Node right) {
            this.split = split;
            this.label = label;
            this.left = left;
            this.right = right;
        }

        // Returns true if and only if this node is a leaf node.
        public boolean isLeaf() {
            return left == null && right == null;
        }
    }
}
