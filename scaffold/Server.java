import java.util.*;
import java.io.*;
import java.net.*;
import java.nio.file.*;

import com.sun.net.httpserver.*;

import org.apache.commons.csv.*;

import smile.data.*;
import smile.io.*;

public class Server {
    // Port number used to connect to this server
    private static final int PORT = Integer.parseInt(System.getenv().getOrDefault("PORT", "8000"));

    public static void main(String[] args) throws IOException, URISyntaxException {
        if (args.length != 1) {
            throw new IllegalArgumentException("java Server [tsv file]");
        }
        String filename = args[0];
        DataFrame df = Read.csv(filename, CSVFormat.TDF.withHeader());
        String[] corpus = df.stringVector("message").toStringArray();
        boolean[] labels = df.booleanVector("label").array();
        TextClassifier clf = TextClassifier.from(corpus, labels);
        HttpServer server = HttpServer.create(new InetSocketAddress(PORT), 0);
        server.createContext("/", (HttpExchange t) -> {
            String html = Files.readString(Paths.get("index.html"));
            send(t, "text/html; charset=utf-8", html);
        });
        server.createContext("/query", (HttpExchange t) -> {
            String s = parse("s", t.getRequestURI().getQuery().split("&"));
            send(t, "application/json", Boolean.toString(clf.classify(s)));
        });
        server.setExecutor(null);
        server.start();
    }

    private static String parse(String key, String... params) {
        for (String param : params) {
            String[] pair = param.split("=");
            if (pair.length == 2 && pair[0].equals(key)) {
                return pair[1];
            }
        }
        return "";
    }

    private static void send(HttpExchange t, String contentType, String data)
            throws IOException, UnsupportedEncodingException {
        t.getResponseHeaders().set("Content-Type", contentType);
        byte[] response = data.getBytes("UTF-8");
        t.sendResponseHeaders(200, response.length);
        try (OutputStream os = t.getResponseBody()) {
            os.write(response);
        }
    }
}
