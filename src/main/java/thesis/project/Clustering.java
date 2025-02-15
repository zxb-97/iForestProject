package thesis.project;

import java.util.*;

public class Clustering {
    public static List<double[]> fft(double[][] data, int K){
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
}
