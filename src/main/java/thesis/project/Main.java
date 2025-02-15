package thesis.project;
import java.io.*;
import java.util.*;
import de.jstacs.classifiers.performanceMeasures.AucROC;
import de.jstacs.classifiers.performanceMeasures.AucPR;
import de.jstacs.results.NumericalResultSet;


public class Main {

    private static void confusionMatrix(int[] predictions, long[] dataY){
        /*
        * Computes confusion matrix, precision, recall, F1-score
        * */
        int trueAnomalies = 0;
        int falseAnomalies = 0;
        int trueNormals = 0;
        int falseNormals = 0;

        for (int i = 0; i < dataY.length; i++) {
            if (dataY[i] == 1 && predictions[i] == 1) {
                trueAnomalies++;
            }
            if (dataY[i] == 0 && predictions[i] == 1) {
                falseAnomalies++;
            }
            if (dataY[i] == 0 && predictions[i] == 0) {
                trueNormals++;
            }
            if (dataY[i] == 1 && predictions[i] == 0) {
                falseNormals++;
            }
        }
        double precision = (double) trueAnomalies /(trueAnomalies+falseAnomalies);
        double recall = (double) trueAnomalies /(trueAnomalies+falseNormals);
        // Print out the confusion matrix
        System.out.println(" ");
        System.out.println("True anomalies: " + trueAnomalies);
        System.out.println("False anomalies: " + falseAnomalies);
        System.out.println("True normals: " + trueNormals);
        System.out.println("False normals: " + falseNormals);
        System.out.println(" ");
        System.out.println("precision: " + precision);
        System.out.println("recall: " + recall);
        System.out.println("F1 score = " + 2 * (precision * recall) / (precision + recall));
        System.out.println(" ");
    }

    private static List<double[]> fft(double[][] data, int K){
        /*
        * Computes the set of K centers implementing FFT algorithm
        * */
        // Set seed for reproducible results
        Random rand = new Random(42);

        List<double[]> centers = new ArrayList<>();

        // Randomly choose first center
        centers.add(data[rand.nextInt(0, data.length)]);

        List<double[]> distances = new ArrayList<>();

        // Initialize list of distances by computing distance between the
        // first center and all other points
        for(int i = 0; i < data.length; i++){
            double dist = eucledeanDistance(centers.get(0),data[i]);
            distances.add(new double[]{i,dist}); // each element is represented as an array [index, distance]
        }

        while(centers.size() < K){
            // Find farthest point from the newly added center
            double[] maxDistElement = distances.stream()
                    .max(Comparator.comparingDouble(entry -> entry[1]))
                    .orElseThrow();
            int farthestPointIndex = (int) maxDistElement[0]; // get the index
            double[] farthestPoint = data[farthestPointIndex]; // find the point given its index

            // Update the list of distances
            for(int i = 0; i < data.length; i++){
                if(!centers.contains(data[i])){
                    double updatedDist = eucledeanDistance(data[i], farthestPoint);
                    distances.get(i)[1] = Math.min(distances.get(i)[1], updatedDist);
                }
            }
            // Once we updated the distances we can add the new center
            centers.add(farthestPoint);
        }
        return centers;
    }

    private static double eucledeanDistance(double[] v1, double[] v2){
        /*
        * Computes the eucledean distance between 2 points
        * */
        double sum = 0;
        for(int i = 0; i < v1.length; i++){
            sum += Math.pow((v1[i] - v2[i]), 2);
        }
        return Math.sqrt(sum);
    }

    private static double[] computeHighestZScores(double[] anomalyScores, int z){
        /*
        * Computes the z points with highest anomaly score
        * */
        double[] highestZScores = new double[z];
        double[] sortedScores = Arrays.stream(anomalyScores).sorted().toArray();

        // Sorted array accessed from right to left since sorted() sorts in ascending order
        int highestZScoresCounter = 0;
        for(int i = sortedScores.length - 1; i > (sortedScores.length - 1) - z; i--){
            highestZScores[highestZScoresCounter] = sortedScores[i];
            highestZScoresCounter++;
        }
        return highestZScores;
    }


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
        confusionMatrix(predictions,dataY);

        List<Double> normalScores = new ArrayList<>();
        List<Double> anomalyScores = new ArrayList<>();
        // Separate normal and anomaly scores
        // Needed for auc computations using jstacs
        for (int i = 0; i < scores.length; i++) {
            if (dataY[i] == 0) { // ground truth
                normalScores.add(scores[i]); // Computed score
            } else {
                anomalyScores.add(scores[i]);
            }
        }

        // Sort scores
        double[] sortedScoresNormals = normalScores.stream().sorted().mapToDouble(Double::doubleValue).toArray();
        double[] sortedScoresAnomalies = anomalyScores.stream().sorted().mapToDouble(Double::doubleValue).toArray();

        // Needed by jstacs, since it's thought for any type of classifier
        // For us its irrelevant therefore all weights are assigned 1.0
        double[] weightsNormals = new double[sortedScoresNormals.length];
        Arrays.fill(weightsNormals, 1.0);
        double[] weightsAnomalies = new double[sortedScoresAnomalies.length];
        Arrays.fill(weightsAnomalies, 1.0);

        // Compute AUC-ROC and AUC-PR
        try {
            AucROC roc = new AucROC();
            AucPR pr = new AucPR();

            NumericalResultSet auroc = roc.compute(sortedScoresAnomalies, weightsAnomalies, sortedScoresNormals, weightsNormals);
            NumericalResultSet aucpr = pr.compute(sortedScoresAnomalies, weightsAnomalies, sortedScoresNormals, weightsNormals);
            if (auroc != null) {
                System.out.println("AUC-ROC: " + auroc.getResultAt(0).getValue());
                System.out.println("AUC-PR (Davis and Goadrich): " + aucpr.getResultAt(0).getValue());
                //System.out.println("AUC-PR (Integral): " + aucpr.getResultAt(1).getValue());
            }

        } catch (Exception e) {
            System.err.println("Error computing AUC-ROC: " + e.getMessage());
        }
    }
}