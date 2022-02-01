'''
Created on Jan 25, 2022
@author: Xingchen Li
'''

import numpy as np
from sklearn import datasets

FEATURE_NAMES=['sepal length', 'sepal width', 'petal length', 'petal width']
TARGET_NAMES = ['setosa', 'versicolor', 'virginica']
# Accuracy of calculation
def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

# Scrambled data set
def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

# Training set test machine segmentation
def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test

# Calculate entropy from label sequence
def calculate_entropy(y):
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * np.log2(p)
    return entropy

# The data is divided into two parts according to characteristics and corresponding thresholds
def divide_on_feature(X, feature_i, threshold):
    mask = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        mask = X[:, feature_i] >= threshold
    else:
        mask = X[:, feature_i] == threshold
    X_1 = X[mask, :]
    X_2 = X[~mask, :]
    return np.array([X_1, X_2])

class DecisionNode():
    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i          # Index value of the feature to be tested on the current node
        self.threshold = threshold          # The test threshold
        self.value = value                  # Predicted results (non-leaf nodes this property is None)
        self.true_branch = true_branch      # The left subtree
        self.false_branch = false_branch    # The right subtree

class DecisionTree(object):
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None, criterion = "info_gain"):
        # The root node of the decision tree
        self.root = None
        # The minimum number of samples that a node needs to split
        self.min_samples_split = min_samples_split
        # Nodes need to split the minimum impurity
        self.min_impurity = min_impurity
        # The maximum depth of the decision tree
        self.max_depth = max_depth
        # Calculate the index of impurity(gini,info_gain,gain_ratio...)
        self.criterion = criterion
        # Calculate the function of the current node impurity according to the impurity index
        self._impurity_calculation = None
        # A function that calculates the value of a leaf node
        self._leaf_value_calculation = None

    def fit(self, X, y, loss=None):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, current_depth=0):
        largest_impurity = 0    # Initialization impurity is 0
        best_criteria = None    # The best way to classify current nodes is by a dictionary　{"feature_i":xxx, "threshold":xxx}
        best_sets = None        # The two subsets are a dictionary　{"leftX":XXX, "lefty":xxx, "rightX":xxx, "righty":xxx}

        # If y is an n-dimensional vector then I'm going to extend it to an n-by-1 matrix
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        # Join X and y
        Xy = np.concatenate((X, y), axis=1)
        # The number of samples and features of the current node
        n_samples, n_features = np.shape(X)

        # Only if the sample size is greater than or equal to the set minimum and the depth is less than or equal to the set maximum depth will the points be split
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Facilitate the calculation of the corresponding impurity for each feature
            for feature_i in range(n_features):
                # Get all the different values for the current feature
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                # Traverse all the different characteristic of current value, treating it as a threshold, the data set is divided into two parts (greater than or equal to | small threshold value and threshold value), and then calculate the corresponding purity
                for threshold in unique_values:
                    # The data set is divided into two parts based on thresholds(Xy1 >= threshold | Xy2 < threshold)
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)
                    # Impurity can be calculated only when the size of the two subsets after partition is greater than 0, otherwise it is meaningless
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # Impurity is calculated from the label values of the original set and the two subsets
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]
                        impurity = self._impurity_calculation(y, y1, y2)
                        # If the current impurity is improved from the previous maximum impurity, the maximum impurity is updated, and best_criteria和best_sets
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],   # X of left subtree
                                "lefty": Xy1[:, n_features:],   # y of left subtree
                                "rightX": Xy2[:, :n_features],  # X of right subtree
                                "righty": Xy2[:, n_features:]   # y of right subtree
                                }
        # Facilitate node splitting if the maximum impurity obtained by all the features and all the corresponding feature segmentation points is greater than the set threshold of impurity
        if largest_impurity > self.min_impurity:
            # Build subtrees for the right and left branches
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                                "threshold"], true_branch=true_branch, false_branch=false_branch)
        # All splitting criteria are not met, the leaves are reached, and the final prediction result is obtained through voting
        leaf_value = self._leaf_value_calculation(y)

        return DecisionNode(value=leaf_value)


    def predict_value(self, x, tree=None):
        if tree is None:
            tree = self.root

        # If the value of the current node is not None, the leaf node is reached
        if tree.value is not None:
            return tree.value
        # The corresponding characteristic values are obtained according to tree.feature_i and tested against the threshold value of the current node to determine which branch to follow
        feature_value = x[tree.feature_i]
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch
        # Make predictions recursively
        return self.predict_value(x, branch)

    def predict(self, X):
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    def print_tree(self, tree=None, indent="-"):
        if not tree:
            tree = self.root
        if tree.value is not None:
            print (TARGET_NAMES[int(tree.value)])
        else:
            print ("[%s]>%s?" % (FEATURE_NAMES[tree.feature_i], tree.threshold))
            # Print the true scenario
            print ("%sleft:" % (indent), end="")
            self.print_tree(tree.true_branch, indent * 3)
            # Print the false scenario
            print ("%sright:" % (indent), end="")
            self.print_tree(tree.false_branch, indent * 3)

class ClassificationTree(DecisionTree):
    # Computational information gain
    def _calculate_information_gain(self, y, y1, y2):
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * \
            calculate_entropy(y1) - (1 - p) * \
            calculate_entropy(y2)
        return info_gain
    # Calculate the information gain rate
    def _calculate_gain_ratio(self, y, y1, y2):
        info_gain = self._calculate_information_gain(y, y1, y2)
        p = len(y1) / len(y)
        intrinsic_value = -p * np.log2(p) - (1-p) * np.log2(1-p)
        return info_gain / intrinsic_value
    # Votes are cast based on tags, and if multiple categories have the same number of votes, the first traversal of the category is returned
    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # Count number of occurences of samples with label
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        assert(self.criterion in ["gain_ratio", "info_gain"])
        if self.criterion == "gain_ratio":
            self._impurity_calculation = self._calculate_information_gain
        else:
            self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)


if __name__ == "__main__":
    print ("-- Classification Tree --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = ClassificationTree(criterion="gain_ratio")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    clf.print_tree()
