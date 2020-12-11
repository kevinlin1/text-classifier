import org.junit.jupiter.api.*;
import org.junit.jupiter.api.function.*;
import org.junit.jupiter.params.*;
import org.junit.jupiter.params.provider.*;
import static org.junit.jupiter.api.Assertions.*;

import static org.hamcrest.core.IsEqual.equalTo;

import java.io.*;
import java.net.*;
import java.util.*;
import java.util.stream.*;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TextClassifierTest {

    public enum Source {
        SPAM("spam.tsv", 5000), TOXIC("toxic.tsv", 5000), TINY("tiny.tsv", 10);

        private final String[] corpus;
        private final TextClassifier clf;
        private final GoodTextClassifier gclf;
        private final String filename;

        private Source(String filename, int N) {
            try (Scanner input = new Scanner(new File(filename))) {
                input.nextLine(); // Skip header
                boolean[] labels = new boolean[N];
                this.corpus = new String[N];
                for (int i = 0; i < N; i += 1) {
                    Scanner line = new Scanner(input.nextLine()).useDelimiter("\t");
                    labels[i] = line.nextBoolean();
                    corpus[i] = line.next();
                }
                Vectorizer vectorizer = new BM25Vectorizer();
                Splitter splitter = new GiniSplitter(vectorizer.fitTransform(corpus), labels);
                this.clf = new TextClassifier(vectorizer, splitter);
                this.gclf = new GoodTextClassifier(vectorizer, splitter);
                this.filename = filename;
            } catch (Exception e) {
                System.exit(1); // Because unit tests won't be able to run
                throw new ExceptionInInitializerError();
            }
        }

        public String toString() {
            return filename;
        }
    }

    @ParameterizedTest
    @DisplayName("classify")
    @Order(1)
    @MethodSource("classifyProvider")
    public void testClassify(Source src, String text) {
        assertEquals(src.gclf.classify(text), src.clf.classify(text));
    }

    static Stream<Arguments> classifyProvider() {
        Stream.Builder<Arguments> result = Stream.builder();
        for (Source src : Source.values()) {
            for (String text : src.corpus) {
                result.add(Arguments.of(src, text));
            }
        }
        return result.build();
    }

    @ParameterizedTest
    @DisplayName("print")
    @Order(2)
    @EnumSource
    @CaptureSystemOutput
    public void testPrint(Source src, CaptureSystemOutput.OutputCapture out) {
        out.expect(equalTo(src.gclf.print()));
        src.clf.print();
    }

    @ParameterizedTest
    @DisplayName("print after prune(10)")
    @Order(3)
    @EnumSource
    @CaptureSystemOutput
    public void testPrintAfterPrune10(Source src, CaptureSystemOutput.OutputCapture out) {
        src.gclf.prune(10);
        out.expect(equalTo(src.gclf.print()));
        src.clf.prune(10);
        src.clf.print();
    }

    @ParameterizedTest
    @DisplayName("print after prune(5)")
    @Order(4)
    @EnumSource
    @CaptureSystemOutput
    public void testPrintAfterPrune5(Source src, CaptureSystemOutput.OutputCapture out) {
        src.gclf.prune(5);
        out.expect(equalTo(src.gclf.print()));
        src.clf.prune(5);
        src.clf.print();
    }
}
