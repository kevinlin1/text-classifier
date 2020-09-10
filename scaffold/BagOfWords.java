import java.util.*;

import smile.nlp.*;

// Bag-of-words text representation that stores term-frequency counts.
public class BagOfWords implements TextTerms {
    private Collection<String> words;
    private Map<String, Integer> counts;
    private int maxtf;

    public BagOfWords(Collection<String> words) {
        this.words = words;
        this.counts = new HashMap<>();
        this.maxtf = 0;
        for (String word : words) {
            int count = tf(word) + 1;
            counts.put(word, count);
            if (count > this.maxtf) {
                this.maxtf = count;
            }
        }
    }

    public int size() {
        return words.size();
    }

    public Collection<String> words() {
        return words;
    }

    public Collection<String> unique() {
        return counts.keySet();
    }

    public int tf(String term) {
        return counts.getOrDefault(term, 0);
    }

    public int maxtf() {
        return maxtf;
    }

    public String toString() {
        return words.toString();
    }

    public boolean equals(Object o) {
        if (this == o) {
            return true;
        } else if (!(o instanceof BagOfWords)) {
            return false;
        }
        BagOfWords other = (BagOfWords) o;
        return this.words.equals(other.words);
    }

    public int hashCode() {
        return counts.keySet().hashCode();
    }
}
