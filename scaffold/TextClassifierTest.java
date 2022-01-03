import org.junit.jupiter.api.*;
import org.junit.jupiter.params.*;
import org.junit.jupiter.params.provider.*;
import static org.junit.jupiter.api.Assertions.*;

import java.io.*;
import java.util.*;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TextClassifierTest {

    public enum Source {
        SPAM("spam.tsv"), TOXIC("toxic.tsv"), TINY("tiny.tsv");

        private final String[] messages;
        private final TextClassifier clf;
        private final String filename;

        private Source(String filename) {
            File file = new File(filename);
            int N = 0;
            try (Scanner input = new Scanner(file)) {
                input.nextLine(); // Skip header
                while (input.hasNextLine()) {
                    N += 1;
                    input.nextLine();
                }
            } catch (IOException e) {
                System.exit(1); // Because unit tests won't be able to run
                throw new ExceptionInInitializerError();
            }
            boolean[] labels = new boolean[N];
            this.messages = new String[N];
            try (Scanner input = new Scanner(file)) {
                input.nextLine(); // Skip header
                for (int i = 0; i < N; i += 1) {
                    Scanner line = new Scanner(input.nextLine()).useDelimiter("\t");
                    labels[i] = line.nextBoolean();
                    this.messages[i] = line.next();
                }
            } catch (IOException e) {
                System.exit(1); // Because unit tests won't be able to run
                throw new ExceptionInInitializerError();
            }
            Vectorizer vectorizer = new Vectorizer();
            Splitter splitter = new GiniSplitter(vectorizer.fitTransform(this.messages), labels);
            this.clf = new TextClassifier(vectorizer, splitter);
            this.filename = filename;
        }

        public String toString() {
            return filename;
        }
    }

    @ParameterizedTest
    @DisplayName("classify")
    @Order(1)
    @EnumSource
    public void testClassify(Source src) throws IOException {
        Scanner input = new Scanner(new File(src.filename + ".test1.txt"));
        String expected = input.useDelimiter("\\A").next();
        StringBuilder builder = new StringBuilder();
        for (String text : src.messages) {
            builder.append(src.clf.classify(text));
            builder.append('\n');
        }
        assertEquals(expected, builder.toString());
    }

    @ParameterizedTest
    @DisplayName("print")
    @Order(2)
    @EnumSource
    public void testPrint(Source src) throws IOException {
        Scanner input = new Scanner(new File(src.filename + ".test2.txt"));
        String expected = input.useDelimiter("\\A").next();
        ByteArrayOutputStream b = new ByteArrayOutputStream();
        PrintStream old = System.out;
        System.setOut(new PrintStream(b));

        src.clf.print();

        System.out.flush();
        System.setOut(old);
        assertEquals(expected, b.toString());
    }

    @ParameterizedTest
    @DisplayName("print after prune(10)")
    @Order(3)
    @EnumSource
    public void testPrintAfterPrune10(Source src) throws IOException {
        Scanner input = new Scanner(new File(src.filename + ".test3.txt"));
        String expected = input.useDelimiter("\\A").next();
        ByteArrayOutputStream b = new ByteArrayOutputStream();
        PrintStream old = System.out;
        System.setOut(new PrintStream(b));

        src.clf.prune(10);
        src.clf.print();

        System.out.flush();
        System.setOut(old);
        assertEquals(expected, b.toString());
    }

    @ParameterizedTest
    @DisplayName("print after prune(5)")
    @Order(4)
    @EnumSource
    public void testPrintAfterPrune5(Source src) throws IOException {
        Scanner input = new Scanner(new File(src.filename + ".test4.txt"));
        String expected = input.useDelimiter("\\A").next();
        ByteArrayOutputStream b = new ByteArrayOutputStream();
        PrintStream old = System.out;
        System.setOut(new PrintStream(b));

        src.clf.prune(5);
        src.clf.print();

        System.out.flush();
        System.setOut(old);
        assertEquals(expected, b.toString());
    }

    // Dump solution class output to text files.
    public static void main(String[] args) throws FileNotFoundException {
        for (Source src : Source.values()) {
            System.setOut(new PrintStream(new FileOutputStream(src.filename + ".test1.txt")));
            for (String text : src.messages) {
                System.out.println(src.clf.classify(text));
            }
            System.out.flush();

            System.setOut(new PrintStream(new FileOutputStream(src.filename + ".test2.txt")));
            src.clf.print();
            System.out.flush();

            System.setOut(new PrintStream(new FileOutputStream(src.filename + ".test3.txt")));
            src.clf.prune(10);
            src.clf.print();
            System.out.flush();

            System.setOut(new PrintStream(new FileOutputStream(src.filename + ".test4.txt")));
            src.clf.prune(5);
            src.clf.print();
            System.out.flush();
        }
    }
}
