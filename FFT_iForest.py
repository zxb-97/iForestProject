import numpy as np
import pandas as pd
import random as rand
from scipy.io import arff
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import sys
import time
import math
import logging

logging.basicConfig(filename='output.log', filemode='w',level=logging.INFO)

class CFactor:
    @classmethod
    def compute(cls, n_elements):
        """
        Compute the average path length of unsuccessful BST search
        c(n) = 2H(n-1) - (2(n-1)/n), where H(n) is the harmonic number
        """
        if n_elements < 2:
            return 0
        if n_elements == 2:
            return 1
            
        # H(n) ≈ ln(n) + 0.5772156649 (Euler's constant)
        h_n_minus_1 = np.log(n_elements - 1) + np.euler_gamma
        c = 2 * h_n_minus_1 - (2 * (n_elements - 1) / n_elements)
        return c


class Clustering:
   
    @staticmethod
    def SequentialFFT(attr_values: np.ndarray, K: int) -> np.ndarray:
        # For 1D array (single attribute)
        attr_values = attr_values.reshape(-1, 1)
    
        centers = np.full((K, attr_values.shape[1]), float("inf"))
        centers[0] = attr_values[np.random.choice(attr_values.shape[0])]
        logging.info(f"Cluster center 0: {centers[0]}")

        distances = np.linalg.norm(attr_values - centers[0], axis=1)

        for j in range(1,K):
            farthest_point_index = np.argmax(distances)
            farthest_point = attr_values[farthest_point_index]

            for i, point in enumerate(attr_values):
                if not np.any(np.isclose(centers, point)):
                    distances[i] = min(distances[i], np.linalg.norm(point - farthest_point))
                
            centers[j] = farthest_point
            logging.info(f"Cluster center {j}: {centers[j]}")
    
        clusters = [[] for _ in range(K)]

        for point in attr_values:
            closest_center_index = np.argmin(np.linalg.norm(point - centers, axis=1))
            clusters[closest_center_index].append(point.flatten()[0])  # Convert back to scalar

        return clusters
    
            

            

class Node:
    def __init__(self, *, X=None, children=None, split_attr=None, 
                 indices=None, curr_depth=0):
        self.X = X
        self.children = [] if children is None else children
        self.split_attr = split_attr
        
        self.indices = indices
        self.curr_depth = curr_depth
        self.size = len(indices) if indices is not None else 0
        self.c_factor = CFactor.compute(self.size) # Put here so it is computed at node creation

    # could be sped up inputing a batch instead of single instance
    # this way we can fully use numpy capabilities
    def path_length(self, x, curr_height, height_limit=6):
        """Calculate the path length for instance x with height limit"""
        # If we've reached the height limit, return current height plus c_factor
        if curr_height >= height_limit:
            return curr_height + self.c_factor
            
        # If this is a leaf node, return current height
        if (not self.children) or (all(child is None for child in self.children)):
            return curr_height
        
        # Continue traversal if we haven't hit the limit
        for child in self.children:
            if child is not None:
                cluster_values = self.X[child.indices, self.split_attr]
                if np.any(np.isclose(x[self.split_attr], cluster_values)):
                    return child.path_length(x, curr_height + 1, height_limit)
                
        return curr_height
    



class IsolationTree:
    def __init__(self, X: np.ndarray, debug=True):
        self.X = X
        self.tree_ = []
        self.debug = debug

    def build_tree(self, indices: np.ndarray, depth=0, parent_size=None):
        indent = "  " * depth if self.debug else ""
        
        if self.debug:
            logging.info(f"\n{indent}Building node at depth {depth}")
            logging.info(f"{indent}Number of samples: {len(indices)}")
            if parent_size is not None:
                logging.info(f"{indent}Parent node size: {parent_size}")


        # Create current node with indices
        curr_node = Node(X=self.X, indices=indices, curr_depth=depth)
        self.tree_.append(curr_node)

        # Base case: leaf node
        if len(curr_node.indices) <= 1:
            if self.debug:
                logging.info(f"{indent}└─ Leaf node reached with {len(curr_node.indices)} elements")
            return curr_node

        # Get the data instances given their row index 
        X_subset = self.X[indices]
        
        # Get split attribute
        Q = X_subset.shape[1]
        curr_node.split_attr = np.random.choice(Q)  # Randomly select a feature for splitting
        
        if self.debug:
            logging.info(f"{indent}Selected split attribute: {curr_node.split_attr}")

        # CLUSTER ATTRIBUTE VALUES
        FFT_sample = X_subset[:, curr_node.split_attr]
        q_clusters = Clustering.SequentialFFT(FFT_sample, K=4)  # List of clusters

        if self.debug:
            logging.info(f"{indent}Clusters created for attribute {curr_node.split_attr}: {q_clusters}")

        for cluster in q_clusters:
            cluster_mask = np.isin(FFT_sample, cluster)
            cluster_indices = indices[cluster_mask]

            if len(cluster_indices) > 0:
                if self.debug:
                    logging.info(f"{indent}Cluster contains {len(cluster_indices)} elements, building subtree...")
                # Recursively build the tree for this cluster
                child_node = self.build_tree(cluster_indices, depth + 1, parent_size=curr_node.size)
                curr_node.children.append(child_node)
            else:
                curr_node.children.append(None)
        
        return curr_node

class IsolationTreeEnsemble:
    def __init__(self, X:np.ndarray, sample_size=256, n_trees=10):
            self.sample_size = sample_size
            self.n_trees = n_trees
            self.forest = [] # better ndarray of objects of size n_trees 
            self.height_limit = np.ceil(np.log2(self.sample_size))
            self.X = X


    def build_forest(self) -> list:
            # I compute all subsamples at once for efficiency
            # Each call of np.random.choice samples rows from the entire dataset X
            # without replacement so in a single sample we cannot sample twice an instance
            all_subsamples = np.array([
                np.random.choice(self.X.shape[0], self.sample_size, replace=False)
                for _ in range(self.n_trees)
            ])

            for sample_indices in all_subsamples:
                tree = IsolationTree(self.X)
                tree.build_tree(sample_indices)# depth=0)
                self.forest.append(tree) 

            return self.forest

        
    def avg_path_length(self) -> np.ndarray:
            avg_path_lengths = np.zeros(len(self.X))
            for idx, x in enumerate(self.X):
                path_lengths = []
                for iTree in self.forest:
                    if isinstance(iTree, IsolationTree):
                        root_node = iTree.tree_[0]
                        if isinstance(root_node, Node):
                            # Pass height_limit to path_length
                            path_len = root_node.path_length(x, curr_height=0, 
                                                        height_limit=self.height_limit)
                            path_lengths.append(path_len)
            
                avg_len = np.mean(path_lengths) if path_lengths else 0
                avg_path_lengths[idx] = avg_len

            return avg_path_lengths

    def anomaly_score(self) -> np.ndarray:
        avg_path_lengths = self.avg_path_length()
        c = CFactor.compute(self.sample_size)
        
        anomaly_scores = np.exp2(-avg_path_lengths/c)
        return anomaly_scores

    def predict_anomalies(self, anomaly_scores: np.ndarray, threshold: float) -> np.ndarray:
        return (anomaly_scores >= threshold)

def main():

    # Load the data
    data, meta = arff.loadarff('C:\\MulcrossDataset.arff')
    df = pd.DataFrame(data)

    # Convert bytes to string and then to numeric where possible
    for col in df.columns:
        if df[col].dtype == object:  # Check if column contains bytes/strings
            # Try to convert bytes to string
            df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        
            # If this is the label column (last column), we'll store it separately before removing
            if col == df.columns[-1]:
                y = df[col].values
                df = df.drop(col, axis=1)

    #X = df.to_numpy().astype(float)

    # Sample 10% of the dataset (for faster testing)
    df_sample = df.sample(frac=0.05, random_state=42)  # Take frac% of the dataset
    y_sample = y[df_sample.index]  # Ensure the labels correspond to the sampled instances

    # Convert the sampled data to a numpy array
    X_sample = df_sample.to_numpy().astype(float)

    # Print info about the processed dataset
    print("Dataset shape:", X_sample.shape)
    print("\nFirst few instances:")
    print(X_sample[:5])
    
    # Run the iForest algorithm on the smaller dataset
    start_time = time.time()
    
    ensemble = IsolationTreeEnsemble(X_sample, sample_size=32, n_trees=10)  
    ensemble.build_forest()
    scores = ensemble.anomaly_score()

    
    threshold = np.percentile(scores, 90)  
    predictions = ensemble.predict_anomalies(scores, threshold)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time taken by iForest implementation: {elapsed_time:.4f} seconds")

    print(f"\nNumber of predicted anomalies: {sum(predictions)}")
    print(f"Total number of instances: {len(predictions)}")
    print(f"Percentage of anomalies: {(sum(predictions)/len(predictions))*100:.2f}%")

    # If you want to evaluate accuracy (since we have true labels)
    true_anomalies = (y_sample == 'Anomaly')
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_anomalies, predictions))
    print("\nClassification Report:")
    print(classification_report(true_anomalies, predictions))

    # --- ROC AUC Calculation ---
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(true_anomalies.astype(int), scores)

    # Calculate the AUC
    roc_auc = auc(fpr, tpr)

    print(f"\nROC AUC: {roc_auc:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    plt.close()
    sys.exit()

if __name__ == "__main__":
    main()



