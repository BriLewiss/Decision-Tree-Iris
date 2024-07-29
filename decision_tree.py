from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

class Node():
  def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):

    # Initialize the decision node's attributes
    self.feature_index = feature_index
    self.threshold = threshold
    self.left = left
    self.right = right
    self.info_gain = info_gain

    # Initialize the leaf node
    self.value = value

class DecisionTree():
  def __init__(self, max_depth=2):
    '''
    Initialize the root of the tree and max depth
    '''
    self.root = None
    self.max_depth = max_depth

  def fit(self, X, y):
    '''
    Fit the decision tree to the training data
    '''
    # Concatenate the features and target into a single dataset
    dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
    # Build the decision tree
    self.root = self.build_tree(dataset)

  def build_tree(self, dataset, curr_depth=0):
    '''
    Recursively build the decision tree
    '''
    X = dataset[:, :-1]  # Features
    y = dataset[:, -1]  # Target
    n_samples, n_features = X.shape

    # Split until stopping conditions are met
    if n_samples >= 2 and curr_depth <= self.max_depth:
      # Find the best split
      best_split = self.get_best_split(dataset, n_samples, n_features)

      # Check if information gain is positive
      if best_split["info_gain"] > 0:
        # recur left
        left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)

        # recur right
        right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)

        # return decision node
        return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["info_gain"])

    # compute leaf node
    leaf_value = self.calculate_leaf_value(y)

    # return leaf node
    return Node(value=leaf_value)

  def get_best_split(self, dataset, n_samples, n_features):
    '''
    Find the best split for the decision tree
    '''
    # Dictionary tos tore the best split
    best_split = {}
    max_info_gain = float("-inf")

    # Loop over all features
    for feature_index in range(n_features):
      feature_values = dataset[:, feature_index]
      possible_thresholds = np.unique(feature_values)

      # Loop over all possible thresholds
      for threshold in possible_thresholds:
        # Split the dataset
        dataset_left, dataset_right = self.split(dataset, feature_index, threshold)

        # Check if childs are not null
        if len(dataset_left) > 0 and len(dataset_right) > 0:
          y = dataset[:, -1]
          left_y = dataset_left[:, -1]
          right_y = dataset_right[:, -1]

          # Calculate the information gain
          curr_info_gain = self.information_gain(y, left_y, right_y, "gini")

          # Update the best split if needed
          if curr_info_gain > max_info_gain:
            best_split["feature_index"] = feature_index
            best_split["threshold"] = threshold
            best_split["dataset_left"] = dataset_left
            best_split["dataset_right"] = dataset_right
            best_split["info_gain"] = curr_info_gain
            max_info_gain = curr_info_gain

    return best_split

  def split(self, dataset, feature_index, threshold):
    '''
    Split the dataset based on the given feature and threshold
    '''
    dataset_left = []
    dataset_right = []

    for row in dataset:
      if row[feature_index] <= threshold:
        dataset_left.append(row)
      else:
        dataset_right.append(row)

    return np.array(dataset_left), np.array(dataset_right)

  def information_gain(self, parent, l_child, r_child, mode="entropy"):
    '''
    Calculate the information gain
    '''
    weight_l = len(l_child) / len(parent)
    weight_r = len(r_child) / len(parent)

    if mode == "gini":
      gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
    else:
      gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))

    return gain

  def entropy(self, y):
    '''
    Calculate the entropy
    '''
    class_labels = np.unique(y)
    entropy = 0

    for cls in class_labels:
      p_cls = len(y[y == cls]) / len(y)
      entropy += -p_cls * np.log2(p_cls)

    return entropy

  def gini_index(self, y):
    '''
    Calculate the gini index
    '''
    class_labels = np.unique(y)
    gini = 0

    for cls in class_labels:
      p_cls = len(y[y == cls]) / len(y)
      gini += p_cls**2

    return 1 - gini

  def calculate_leaf_value(self, y):
    '''
    Calculate the leaf node value
    '''
    y = list(y)
    return max(set(y), key=y.count)

  def predict(self, X):
    '''
    Predict the new dataset
    '''
    predictions = []

    for x in X:
      prediction = self.make_prediction(x, self.root)
      predictions.append(prediction)

    return predictions

  def make_prediction(self, x, tree):
    '''
    Predict a single data point
    '''
    if tree.value != None:
      return tree.value

    feature_val = x[tree.feature_index]

    if feature_val <= tree.threshold:
      return self.make_prediction(x, tree.left)
    else:
      return self.make_prediction(x, tree.right)

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

classifier = DecisionTree()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("---------- ACCURACY SCORE ----------")
print(accuracy_score(y_test, y_pred) * 100)
