package thesis.project;

import java.util.Random;

public class TestData {
    private double[][] dataX;
    private long[] dataY;
    private final int numNormal;
    private final int numAnom;
    private final int numNormCluster1;
    private final Random rand;

    public TestData(int numNormal, int numAnom, int numNormCluster1) {
        this.numNormal = numNormal;
        this.numAnom = numAnom;
        this.numNormCluster1 = numNormCluster1;

        dataX = new double[numNormal+numAnom][2];
        dataY = new long[numNormal+numAnom];
        rand = new Random();

    }


    public double[][] createDataX() {
        int cols = 2;
        int rows = numNormal + numAnom;

        // Create Normal points
        for (int i = 0; i < numNormal; i++) { // 0 to 949
            double[] point = new double[cols];
            if (i < numNormCluster1) { // 0 to 349
                for (int j = 0; j < cols; j++) {
                    point[j] = 0.8 + (rand.nextDouble() * 1.2); // 0.8 to 2.0

                }
                dataX[i] = point;
            } else if (i < numNormal) { // 350 to 949
                for (int j = 0; j < cols; j++) {
                    point[j] = 4.0 + rand.nextDouble() * 2.0; // 4.0 to 6.0
                }
                dataX[i] = point;
            }
        }
        // Create anomalies
        for (int i = numNormal ; i < rows; i++) { // 950 to 999
            double[] point = new double[cols];
            for(int j = 0; j < cols; j++){
                point[j] = 12 + rand.nextDouble() * 3; // 12.0 to 15.0
            }
            dataX[i] = point;
        }
        return dataX;
    }
    public long[] createDataY(){
        for( int i = 0; i < numNormal; i++ ){
            dataY[i] = 0;
        }
        for( int i = numNormal; i < (numNormal+numAnom); i++ ){
            dataY[i] = 1;
        }
        return dataY;
    }
}
