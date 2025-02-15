package thesis.project;
import java.io.*;

public class Main {

    public static void main(String[] args) throws IOException {
        //Handle npz data and header
        NpzParser npzParser = new NpzParser("src/main/resources/16_http.npz");

        //Map<String,String> metadataX = NpzHeaderParser.npzHeaderReader(npzParser.getXInputStream());
        //System.out.println(metadataX);

        double[][] data = npzParser.parseX(); // Dataset X.npy converted to java array
        long[] dataY = npzParser.parseY(); // Labels y.npy converted to java array

        // Isolation forest parameter setting
        int sampleSize = 256;
        int nTrees = 100;
        int heightLimit = 8;

        // Initialize IsolationForest object
        IsolationForest iForest = new IsolationForest(data,sampleSize,nTrees, heightLimit);
        // Build the isolation forest
        iForest.buildForest();
        // Compute anomaly scores
        double[] scores = iForest.anomalyScore();


        /// Model Evaluation
        // 10% contamination level for all datasets
        int[] predictions = iForest.predictWithThreshold(0.90, scores);

        // Compute confusion matrix and print it to console
        ModelEvaluation.confusionMatrix(predictions,dataY);
        // Compute AUCROC and AUCPR and print it them console
        ModelEvaluation.computeAUCScores(scores,dataY);


    }
}