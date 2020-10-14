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

import org.apache.commons.csv.*;

import smile.data.*;
import smile.io.*;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TextClassifierTest {

    public enum Source {
        SPAM("spam.tsv", CSVFormat.TDF.withHeader()),
        TOXIC("toxic.tsv", CSVFormat.TDF.withHeader()),
        TINY("tiny.tsv", CSVFormat.TDF.withHeader());

        private final String[] corpus;
        private final TextClassifier clf;
        private final GoodTextClassifier gclf;
        private final String filename;

        private Source(String filename, CSVFormat format) {
            try {
                DataFrame df = Read.csv(filename, format);
                this.corpus = df.stringVector("message").toStringArray();
                boolean[] labels = df.booleanVector("label").array();
                Vectorizer vectorizer = new BM25Vectorizer();
                Splitter splitter = new Splitter(vectorizer.fitTransform(corpus), labels);
                this.clf = new TextClassifier(vectorizer, splitter);
                this.gclf = new GoodTextClassifier(vectorizer, splitter);
                this.filename = filename;
            } catch (IOException | URISyntaxException e) {
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
    public void testPrintAfterPrune(Source src, CaptureSystemOutput.OutputCapture out) {
        src.gclf.prune(10);
        out.expect(equalTo(src.gclf.print()));
        src.clf.prune(10);
        src.clf.print();
    }
}
