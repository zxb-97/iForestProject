package thesis.project;
import java.util.*;

public class IsolationForest {
    private final double[][] X;
    private final int sample_size;
    private final int n_trees;
    private final Node[] forest;
    private final Random rand;
    private final int heightLimit;

    public IsolationForest(double[][] X, int sample_size, int n_trees, int heightLimit) {
        this.X = X;
        this.sample_size = Math.min(sample_size, X.length);
        this.n_trees = n_trees;
        this.forest = new Node[n_trees];
        this.heightLimit = heightLimit;
        this.rand = new Random(42);
    }


    // Sampling without replacement
    public void buildForest() {
        for (int i = 0; i < n_trees; i++) {
            //System.out.println("Building tree:" + i);
            int[] randIndices;
            // I use a Set to enforce sampling without replacement
            Set<Integer> randSet = new HashSet<>();

            // Compute set of (distinct) indices
            while(randSet.size() < sample_size) {
                randSet.add(rand.nextInt(X.length));
            }
            // Convert it to array of ints
            randIndices = randSet.stream().mapToInt(Integer::intValue).toArray();
            // Start building tree i
            forest[i] = buildTree(randIndices, 0);
        }
    }

    private Node buildTree(int[] indices, int depth) {
        /*
        * The tree is built following the paper "Isolation Forest" not "Isolation-based Anomaly Detection"
        * The difference is that in "Isolation Forest" the tree is built until the height limit is reached,
        * while in "Isolation-based Anomaly Detection", the tree is built until all instances are isolated
        * and the height limit is used in the Evaluation stage (path length computation).
        * This choice avoids a StackOverflow error for large datasets with many attributes.
        * */

        //System.out.println("Building node at depth:" + depth);
        //System.out.println("Current sample size: " + indices.length);

        // Sample size and cFactor(sampleSize) initialized in the Node constructor
        Node currNode = new Node(indices, depth);

        //System.out.println("current cFactor:" + currNode.cFactor);

        // Check recursion base case
        if (depth >= heightLimit || indices.length <= 1) {
            //System.out.println("Node is a leaf (sample size <= 1)");
            return currNode;
        }

        // Randomly choose an attribute
        currNode.splitAttr = rand.nextInt(X[0].length);
        //System.out.println("splitAttr = " + currNode.splitAttr);

        // Get the values of the sample elements, for the selected attribute
        double[] splitAttrValues = new double[indices.length];
        for (int i = 0; i < indices.length; i++) {
            splitAttrValues[i] = X[indices[i]][currNode.splitAttr];
        }

        // Compute min and max value to then select minValue <= splitValue <= maxValue
        double minValue = splitAttrValues[0];
        double maxValue = splitAttrValues[0];
        for (double val : splitAttrValues) {
            minValue = Math.min(minValue, val);
            maxValue = Math.max(maxValue, val);
        }
        //System.out.println("minValue = " + minValue);
        //System.out.println("maxValue = "+ maxValue);

        // In the paper authors say that the algorithm should stop
        // if an instance is isolated, OR if all instances inside a node are identical
        // if minValue == maxValue, there's a chance we are in the latter case
        if (minValue == maxValue) {
            boolean allEqual = true;
            // Skip the first value, since it will be equal to itself
            for(int i = 1; i < indices.length; i++){
                if(!Arrays.equals(X[indices[0]], X[indices[i]])){
                    allEqual = false;
                    break; // Stop iterating and initialize currNode.splitValue
                }
            }// here all instances are equal therefore exit
            if(allEqual){
                return currNode;
            }
        }


        currNode.splitValue = (rand.nextDouble() * (maxValue - minValue)) + minValue;
        //System.out.println("splitValue = " + currNode.splitValue);

        ArrayList<Integer> leftIndices = new ArrayList<>();
        ArrayList<Integer> rightIndices = new ArrayList<>();

        // Compute indices for left and right child nodes
        for (int idx : indices) {
            if (X[idx][currNode.splitAttr] < currNode.splitValue) {
                leftIndices.add(idx);
            } else {
                rightIndices.add(idx);
            }
        }

        int[] leftArray = leftIndices.stream().mapToInt(i -> i).toArray();
        int[] rightArray = rightIndices.stream().mapToInt(i -> i).toArray();
        // Recursive call to produce children
        currNode.leftChild = buildTree(leftArray, depth + 1);
        currNode.rightChild = buildTree(rightArray, depth + 1);

        return currNode;
    }



    private double[] avgPathLengths() {
        double[] pathLengths = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            double pathSum = 0;
            for (Node root : forest) {
                // Sum path lengths for instance X[i] across all trees
                pathSum += pathLength(root, X[i],0);
            }
            // Compute the average path length for an instance X[i] and store it
            // in the pathLengths array
            pathLengths[i] = pathSum / n_trees;
        }
        return pathLengths;
    }

    public double[] anomalyScore(){
        double[] anomScores = new double[X.length];
        // Compute average path length for all instances
        double[] averagePathLengths = avgPathLengths();
        double c = cFactor(sample_size);
        // Anomaly score computation for all instances
        for(int i = 0; i < X.length; i++){
            anomScores[i] = Math.pow(2, -averagePathLengths[i]/c);
        }
        return anomScores;
    }


    public int[] predictWithThreshold(double percentile, double[] rawScores) {
        /*
        * Labels all instances given the threshold (which depends on percentile)
        * */
        double threshold = computeThreshold(rawScores, percentile);
        System.out.println("Using threshold: " + threshold);

        int[] predictions = new int[rawScores.length];
        for (int i = 0; i < rawScores.length; i++) {
            predictions[i] = rawScores[i] < threshold ? 0 : 1;
        }
        return predictions;
    }

    public double computeThreshold(double[] rawScores, double percentile) {
        /*
        * Computes the threshold given the specified percentile
        * */
        int n = rawScores.length;
        double[] sortedScores = Arrays.copyOf(rawScores, n);
        Arrays.sort(sortedScores);  // Sort scores in ascending order

        int index = (int) Math.ceil(percentile * n) - 1;  // Find the index for the percentile
        index = Math.max(0, Math.min(index, n - 1));  // Ensure valid index bounds

        return sortedScores[index];
    }


    private double pathLength(Node node, double[] x, int height) {
        // Base case: If the node is a leaf (both children are null or height limit reached)
        // Height limit not checked because tree construction stops at height limit
        if (node.leftChild == null && node.rightChild == null) {
            return height + cFactor(node.currSampleSize);  // Add cFactor for leaf node
        }
        // Otherwise, continue traversing the tree
        if (x[node.splitAttr] < node.splitValue) {
            // If the value of x at the split attribute is less than the split value, go left
            return pathLength(node.leftChild, x, height + 1);

        } else {
            // Otherwise, go right
            return pathLength(node.rightChild, x, height + 1);
        }
    }

    private static double cFactor(int n) {
        final double EULER = 0.5772156649;
        if (n < 2) return 0;
        if (n == 2) return 1;
        double h = Math.log(n - 1) + EULER;
        return 2 * h - (2.0 * (n - 1) / n);
    }

    private static class Node {
        Node leftChild;
        Node rightChild;
        int splitAttr;
        double splitValue;
        final int[] indices;
        int currDepth;
        int currSampleSize;
        double cFactor;

        Node(int[] indices, int currDepth) {
            this.indices = indices;
            this.currDepth = currDepth;
            this.currSampleSize = indices.length;
            this.cFactor = cFactor(this.currSampleSize);
        }
    }
}