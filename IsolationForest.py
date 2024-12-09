import numpy as np
import pandas as pd
import random as rand
from scipy.io import arff
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import sys
import time
import math
import requests
import io

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

class Node:
    def __init__(self, *, left_child=None, right_child=None, split_attr=None, 
                 split_value=None, indices=None, curr_depth=0):
        self.left_child = left_child
        self.right_child = right_child
        self.split_attr = split_attr
        self.split_value = split_value
        self.indices = indices
        self.curr_depth = curr_depth
        self.size = len(indices) if indices is not None else 0
        self.c_factor = CFactor.compute(self.size) # Put here so it is computed at node creation

    # Could be sped up inputing a batch instead of single instance
    def path_length(self, x, curr_height, height_limit=float('inf')):
        """Calculate the path length for instance x with height limit"""
        # If we've reached the height limit, return current height + c_factor
        if curr_height >= height_limit:
            return curr_height + self.c_factor
            
        # If this is a leaf node, return current height
        if (self.left_child is None) and (self.right_child is None):
            return curr_height
        
        # Continue traversal if we haven't hit the limit
        if (x[self.split_attr] < self.split_value) and (isinstance(self.left_child, Node)):
            return self.left_child.path_length(x, curr_height + 1, height_limit)
        elif (isinstance(self.right_child, Node)):
            return self.right_child.path_length(x, curr_height + 1, height_limit)
        else:
            return curr_height

class IsolationTree:
    def __init__(self, X: np.ndarray, debug=False):
        self.X = X
        self.tree_ = []
        self.debug = debug

    def build_tree(self, indices: np.ndarray, depth=0):
        indent = "  " * depth if self.debug else ""
        
        if self.debug:
            print(f"\n{indent}Building node at depth {depth}")
            print(f"{indent}Number of samples: {len(indices)}")

        # Create current node. Contains the (row) indices of instances in the current partition
        curr_node = Node(indices=indices, curr_depth=depth)
        self.tree_.append(curr_node)

        # Base case: leaf node
        if len(curr_node.indices) <= 1:
            if self.debug:
                print(f"{indent}└─ Leaf node reached with {len(curr_node.indices)} elements")
            return curr_node

        # Get the subset of data for current node
        X_subset = self.X[indices]
        
        # Select split attribute
        Q = X_subset.shape[1]
        curr_node.split_attr = np.random.choice((Q)) 

        # Get attribute's range 
        min_value = np.min(self.X[indices, curr_node.split_attr])
        max_value = np.max(self.X[indices, curr_node.split_attr])

        if self.debug:
            print(f"{indent}├─ Selected attribute: {curr_node.split_attr}")
            print(f"{indent}├─ Value range: [{min_value:.2f}, {max_value:.2f}]")

        # Handle end case where all values are identical
        if min_value == max_value:
            if np.all(np.all(X_subset == X_subset[0,:], axis=0)):
                if self.debug:
                    print(f"{indent}└─ All samples identical, stopping split")
                curr_node.split_attr = None
                curr_node.left_child = None
                curr_node.right_child = None
                return curr_node
            
            curr_node.split_value = min_value
        else:# Here not all values are identical
            curr_node.split_value = np.random.uniform(min_value, max_value) 

        if self.debug:
            print(f"{indent}├─ Split value: {curr_node.split_value:.2f}")

        # Partition the indices based on the split value
        left_mask = X_subset[:, curr_node.split_attr] < curr_node.split_value
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        if self.debug:
            print(f"{indent}├─ Left split size: {len(left_indices)}")
            print(f"{indent}└─ Right split size: {len(right_indices)}")

        # Recurse on children
        curr_node.left_child = self.build_tree(left_indices, depth + 1) if len(left_indices) > 0 else None
        curr_node.right_child = self.build_tree(right_indices, depth + 1) if len(right_indices) > 0 else None

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
        # Each call of np.random.choice, samples rows from the entire dataset X
        # without replacement so in a sample we cannot select twice an instance
        all_subsamples = np.array([
            np.random.choice(self.X.shape[0], self.sample_size, replace=False)
            for _ in range(self.n_trees)
        ])

        for sample_indices in all_subsamples:
            tree = IsolationTree(self.X)
            tree.build_tree(sample_indices)
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
    # Mulcross dataset has 5 attributes, 4 numerical , the fifth is either "Normal" or "Anomaly"
    
    #--- Dataset needs to be in same directory as IsolationForest.py ---
    data, meta = arff.loadarff('C:\\MulcrossDataset.arff')
    df = pd.DataFrame(data)

    # Separate numerical attributes from labels 
    for col in df.columns:
        if df[col].dtype == object:  
            df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        
            # If this is the label column (last column), we'll store it separately before removing
            if col == df.columns[-1]:
                y = df[col].values
                df = df.drop(col, axis=1)

    
    # Convert to numpy array
    X = df.to_numpy().astype(float)
    

    # Print info about the processed dataset
    print("Dataset shape:", X.shape)
    print("\nFirst few instances:")
    print(X[:5])
    

    # Now X contains only the feature columns as float values
    # and y contains the labels for evaluation
    start_time = time.time()
    
    ensemble = IsolationTreeEnsemble(X, sample_size=32, n_trees=50)  
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

    # To evaluate accuracy (since we have true labels)
    true_anomalies = (y == 'Anomaly')
    
    conf_matrix = confusion_matrix(true_anomalies, predictions)
    print("\nConfusion Matrix:")
    print(f"True Negatives (TN): {conf_matrix[0, 0]} - Normal instances correctly classified as normal.")
    print(f"False Positives (FP): {conf_matrix[0, 1]} - Normal instances incorrectly classified as anomalies.")
    print(f"False Negatives (FN): {conf_matrix[1, 0]} - Anomalies incorrectly classified as normal.")
    print(f"True Positives (TP): {conf_matrix[1, 1]} - Anomalies correctly classified as anomalies.")
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





