package thesis.project;

import de.jstacs.classifiers.performanceMeasures.AucPR;
import de.jstacs.classifiers.performanceMeasures.AucROC;
import de.jstacs.results.NumericalResultSet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ModelEvaluation {
    public static void computeAUCScores(double[] scores, long[] dataY){
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


    public static void confusionMatrix(int[] predictions, long[] dataY){
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
}
