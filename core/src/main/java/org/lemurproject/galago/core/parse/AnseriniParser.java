// BSD License (http://lemurproject.org/galago-license)
package org.lemurproject.galago.core.parse;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.lemurproject.galago.core.types.DocumentSplit;
import org.lemurproject.galago.utility.Parameters;

import java.io.BufferedReader;
import java.io.IOException;

public class AnseriniParser extends DocumentStreamParser {

    BufferedReader reader;
    JSONParser jsonParser = new JSONParser();

    public AnseriniParser(DocumentSplit split, Parameters p) throws IOException {
        super(split, p);
        this.reader = getBufferedReader(split);
    }

    public Document nextDocument() throws IOException {
        // entire document exists on a single line
        String line;
        while ((line = reader.readLine()) != null) {
            JSONObject jsonObject = null;
            try {
                jsonObject = (JSONObject) jsonParser.parse(line);
            } catch (ParseException e) {
                System.out.println("JSON exception parsing line");
            }
            if (!jsonObject.containsKey("id")) {
                System.out.println("No id in line");
            }
            if (!jsonObject.containsKey("contents")) {
                System.out.println("No contents in line");
            }
            String docid = (String) jsonObject.get("id");
            String text = (String) jsonObject.get("contents");
            Document res = new Document(docid, text);
            return res;
        }
        return null;
    }

    @Override
    public void close() throws IOException {
        if (reader != null) {
            this.reader.close();
            reader = null;
        }
    }
}
