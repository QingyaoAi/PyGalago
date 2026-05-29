package org.lemurproject.galago.core.tools.apps;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.lemurproject.galago.core.retrieval.RetrievalFactory;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.Results;
import org.lemurproject.galago.core.retrieval.ScoredDocument;
import org.lemurproject.galago.core.parse.Document;
import org.lemurproject.galago.core.retrieval.query.StructuredQuery;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.utility.Parameters;
import org.lemurproject.galago.utility.json.JSONUtil;
import org.lemurproject.galago.utility.queries.JSONQueryFormat;
import org.lemurproject.galago.utility.tools.AppFunction;
import org.lemurproject.galago.utility.tools.Arguments;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

public class BatchSearchWithReranker extends AppFunction {

    public static final Logger logger = Logger.getLogger("BatchSearchWithReranker");

    public static void main(String[] args) throws Exception {
        (new BatchSearch()).run(Arguments.parse(args), System.out);
    }

    @Override
    public String getName() {
        return "batch-search-with-reranker";
    }

    @Override
    public String getHelpString() {
        return "galago batch-search-with-reranker <args>\n\n"
                + "  Same as batch-search, but also calls a user-provided reranker to refine each query's results.\n"
                + "  See the documentation for details about how to write the reranker program.\n"
                + "  Produces TREC-formatted output.\n"
                + "  The output can be used with retrieval evaluation tools like\n"
                + "  galago eval (org.lemurproject.galago.core.eval).\n\n"
                + "  The arguments are the same as for batch-search with the addition of two arguments:\n"
                + "     rerankerProgramPath and rerankerCommand (see below).\n"
                + "  Sample invocation:\n"
                + "     galago batch-search-with-reranker --index=/tmp/myindex --requested=200 \\ \n"
                + "          --outputFile=query1.out --systemName=WITH_RERANKING \\ \n"
                + "          --rerankerProgramPath=/mnt/rerankers --rerankerCommand 'python3 bert_reranker.py' \\ \n"
                + "          /mnt/queries/query1.json\n\n"
                + "  Args:\n"
                + "     --index=path_to_your_index\n"
                + "     --requested=N               : Number of results to return for each query.  default=1000\n"
                + "     --outputFile=pathname       : Write the output to this file instead of to STDOUT.\n"
                + "     --verbose=true|false        : If verbose is true, print each query's number, text, and \n"
                + "                                   transformed text as it is processed.  default=false\n"
                + "     --appendFile=true|false     : Use append mode when writing to the output file.  default=false\n"
                + "     --systemName=name           : Use this as the system name in the TREC output.  default=galago\n"
                + "     --operatorWrap=operator     : Wrap query text in the specified operator.\n"
                + "     --queryFormat=json|tsv      : Accept query file in JSON or TSV format.  default=json\n"
                + "     --showNoResults=true|false  : Print dummy result for queries with no results.\n"
                + "                                   This ensures query evaluation metrics account for queries\n"
                + "                                   that returned no results rather than skipping them.\n"
                + "                                   Dummy doc will look like the following\n"
                + "                                   <qid> Q0 no_results_found 1 -999.9 galago \n"
                + "                                   default=false\n"
                + "     --systemName=system_label   : A run label added to a results list queries.  Only available\n"
                + "                                   in trec mode (--trec=true).  default=galago\n"
                + "     --rerankerProgramPath=pathname  : The directory that contains\n"
                + "                                   your reranker program. Your program will run in this directory.\n"
                + "     --rerankerCommand=cmd       : The command line that invokes your reranker program.\n"
                + "                                   For example, 'CUDA_VISIBLE_DEVICES=3 python3 rerank_it.py'\n"
                + "     /path/to/query/file.json    : Input query file in JSON or TSV format (see below).\n\n"

                + "  Query file format:\n"
                + "    The query file is an JSON file containing a set of queries.  Each query\n"
                + "    has text field, which contains the text of the query, and a number field, \n"
                + "    which uniquely identifies the query in the output.\n\n"
                + "  Example query file:\n"
                + "  {\n"
                + "     \"queries\" : [\n"
                + "       {\n"
                + "         \"number\" : \"CACM-408\", \n"
                + "         \"text\" : \"#combine(my query)\"\n"
                + "       },\n"
                + "       {\n"
                + "         \"number\" : \"WIKI-410\", \n"
                + "         \"text\" : \"#combine(another query)\" \n"
                + "       }\n"
                + "    ]\n"
                + "  }\n";
    }

    @Override
    public void run(Parameters parameters, PrintStream out) throws Exception {

        if (!parameters.containsKey("rerankerProgramPath")
                || !parameters.containsKey("rerankerCommand")) {
            out.println(this.getHelpString());
            return;
        }

        if (!(parameters.containsKey("query") || parameters.containsKey("queries"))) {
            out.println(this.getHelpString());
            return;
        }

        // ensure we can print to a file instead of the commandline
        if (parameters.isString("outputFile")) {
            boolean append = parameters.get("appendFile", false);
            out = new PrintStream(new BufferedOutputStream(
                    new FileOutputStream(parameters.getString("outputFile"), append)), true, "UTF-8");
        }

        //- Do we show a no result query dummy doc in output?
        boolean showNoResults = false;
        if (parameters.containsKey ("showNoResults")) {
            showNoResults = parameters.getBoolean ("showNoResults");
        }

        //- Set a system name for the query submissions
        String sysName = parameters.get ("systemName", "galago");

        // get queries
        List<Parameters> queries;
        String queryFormat = parameters.get("queryFormat", "json").toLowerCase();
        switch (queryFormat)
        {
            case "json":
                queries = JSONQueryFormat.collectQueries(parameters);
                break;
            case "tsv":
                queries = JSONQueryFormat.collectTSVQueries(parameters);
                break;
            default: throw new IllegalArgumentException("Unknown queryFormat: "+queryFormat+" try one of JSON, TSV");
        }

        // open index
        Retrieval retrieval = RetrievalFactory.create(parameters);

        // record results requested
        int requested = (int) parameters.get("requested", 1000);

        // for each query, run it, get the results, rerank them, print in TREC format

        for (Parameters query : queries) {
            String queryText = query.getString("text");
            String queryNumber = query.getString("number");

            query.setBackoff(parameters);
            query.set("requested", requested);

            if (parameters.get("verbose", false)) {
                logger.info("RUNNING: " + queryNumber + " : " + queryText);
            }

            // parse and transform query into runnable form
            Node root = StructuredQuery.parse(queryText);

            // --operatorWrap=sdm will now #sdm(...text... here)
            if(parameters.isString("operatorWrap")) {
                if(root.getOperator().equals("root")) {
                    root.setOperator(parameters.getString("operatorWrap"));
                } else {
                    Node oldRoot = root;
                    root = new Node(parameters.getString("operatorWrap"));
                    root.add(oldRoot);
                }
            }
            Node transformed = retrieval.transformQuery(root, query);

            if (parameters.get("verbose", false)) {
                logger.info("Transformed Query:\n" + transformed.toPrettyString());
            }

            // run query
            Results originalResults = retrieval.executeQuery(transformed, query);

            // rerank the results
            List<ScoredDocument> rerankedResults = rerank(queryNumber, queryText, sysName, originalResults,
                    parameters.getString("rerankerProgramPath"),
                    parameters.getString("rerankerCommand"));

            if (!rerankedResults.isEmpty()) {
                for (ScoredDocument sd : rerankedResults) {
                    out.println (sd.toTRECformat (queryNumber, sysName));
                }
            }
            // Even if no results, print SOMETHING so we know.  Evaluation metrics
            // get thrown off when a query is unaccounted for in a ranked list because
            // nothing was retrieved.  Print dummy document output.
            else {
                if (showNoResults) {
                    ScoredDocument sd = new ScoredDocument ();
                    sd.score = -999;
                    sd.rank = 1;
                    sd.documentName = "no_results_found";
                    out.println (sd.toTRECformat (queryNumber, sysName));
                }
            }
        }

        if (parameters.isString("outputFile")) {
            out.close();
        }
    }

    private void callReranker(String queryInfoFileName, String rerankedQueryInfoFileName,
                              String programPath, String programToRun) {
        /* cd to their program directory so they can assume it is running there */
        programToRun = "cd " + programPath + " && " + programToRun;
        /* Leave STDOUT and STDERR from reranker in a file so they can debug */
        String rerankerOutputFilePath = programPath + "/galago_reranker.out";
        String cmd = programToRun + " "
                + programPath + " "
                + queryInfoFileName + " "
                + rerankedQueryInfoFileName
                + " >& " + rerankerOutputFilePath;
        System.out.println ("Reranker is executing this command: " + cmd);

        try {
            Files.delete(Paths.get(rerankerOutputFilePath));
        } catch (IOException ignore) {
            ;
        }

        int exitVal = 0;
        try {
            ProcessBuilder processBuilder = new ProcessBuilder();
            processBuilder.command("bash", "-c", cmd);
            Process process = processBuilder.start();

            exitVal = process.waitFor();
        } catch (Exception cause) {
            System.out.println ("Exception from reranker process: " + cause.getMessage());
            System.exit (1);
        }
        if (exitVal != 0) {
            System.out.println("Unexpected exit value from reranker: " + exitVal);
            System.exit (1);
        }
    }

    private List<ScoredDocument> rerank (String queryNumber, String qText, String sysName, Results originalResults,
                                         String programPath, String programToRun) {
        String queryInfoFileName = "galago_query_info.json";  // they don't get to choose this name
        String rerankedQueryInfoFileName = "galago_reranked_query_info.json"; // they don't get to choose this name
        String queryInfoFilePath = programPath + "/" + queryInfoFileName;
        String rerankedQueryInfoFilePath = programPath + "/" + rerankedQueryInfoFileName;

        List<ScoredDocument> outputResults = null;

        try {
            String jsonText = buildQueryInfoJSON(originalResults, queryNumber, qText, sysName);
            PrintStream out = new PrintStream(new BufferedOutputStream(
                    new FileOutputStream(queryInfoFilePath, false)), true, "UTF-8");
            out.println(jsonText);

            callReranker(queryInfoFileName, rerankedQueryInfoFileName, programPath, programToRun);

            outputResults = getScoredDocuments(rerankedQueryInfoFilePath);

        } catch (Exception ex) {
            System.out.println("Exception: " + ex.toString());
            System.exit (1);
        }
        return outputResults;
    }

    private List<ScoredDocument> getScoredDocuments(String rerankedQueryInfoFilePath) {
        List<ScoredDocument> hits = new ArrayList<>();
        try {
            File tempFile = new File(rerankedQueryInfoFilePath);
            if (!tempFile.exists()) {
                System.out.println("Reranked query info file does not exist: " + rerankedQueryInfoFilePath);
                System.exit(1);
            }
            Reader reader = new BufferedReader(new InputStreamReader(
                    new FileInputStream(rerankedQueryInfoFilePath)));
            JSONParser parser = new JSONParser();
            JSONObject head = (JSONObject) parser.parse(reader);
            JSONArray results = (JSONArray) head.get("results");
            for (Object oHit : results) {
                JSONObject r = (JSONObject) oHit;
                String docid = (String) r.get("docid");
                int rank = ((Long) r.get("rank")).intValue();
                double score = (double) r.get("score");
                ScoredDocument scoredDocument = new ScoredDocument(docid, rank, score);
                hits.add(scoredDocument);
            }
        } catch (Exception ex) {
            System.out.println ("Exception in reranker while reading reranked query info file "
                    + rerankedQueryInfoFilePath);
            System.out.println(ex.getMessage());
            System.exit (1);
        }
        return hits;
    }

    private String buildQueryInfoJSON(Results originalResults, String queryNumber, String qText, String sysName) {
        StringBuilder builder = new StringBuilder();
        try {
            Map<String, Document> docs = originalResults.pullDocuments(Document.DocumentComponents.All);
            builder.append("{ \"queryNumber\": \"");
            builder.append(JSONUtil.escape(queryNumber));
            builder.append("\", \"sysName\": \"");
            builder.append(JSONUtil.escape(sysName));
            builder.append("\", \"query\": \"");
            builder.append(JSONUtil.escape(qText));
            builder.append("\", \n\"results\": [");

            int docIndex = 0;
            for (ScoredDocument sd : originalResults.scoredDocuments) {
                ++docIndex;
                Document document = docs.get(sd.documentName);
                String doctext = document.text;
                int rank = sd.rank;
                double score = sd.score;
                String eid = sd.documentName; // external ID

                builder.append("\n{ \"docid\": \"" + JSONUtil.escape(eid) + "\", "
                        + "\"rank\": " + rank + ","
                        + "\"score\": " + score + ","
                        + "\"doctext\": \"" + JSONUtil.escape(doctext) + "\"}");
                if (docIndex < originalResults.scoredDocuments.size()) {
                    builder.append(",");
                }
            }
            builder.append("]}");
        } catch (Exception ex) {
            System.out.println("Exception: " + ex.toString());
        }
        return builder.toString();
    }

}
