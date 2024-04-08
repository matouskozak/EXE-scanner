import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from pefeatures import PEFeatureExtractor

feature_extractor = PEFeatureExtractor(2)

class GBDTMalwareClassifier:
    """
    Gradient Boosting Decision Tree (GBDT) Malware Classifier.

    This class provides methods for training and using a GBDT model for malware classification.

    Attributes:
        model (lgb.Booster): The trained GBDT model.
        threshold (float): The classification threshold.
        _params (dict): Parameters for the GBDT model.
        fpr (float): The desired false positive rate (FPR).

    Methods:
        predict(X): Predicts the class labels for the given feature vectors.
        predict_proba(X): Predicts the class probabilities for the given feature vectors.
        get_score(bytez): Computes the malware score for the given raw bytes.
        get_label(bytez): Predicts the class label for the given raw bytes.
        print_info(): Prints information about the model.
        train(X_train, y_train, X_val, y_val, FPR): Trains the GBDT model.
        load(model_path, roc_curve_path): Loads a pre-trained GBDT model.
        save(save_path): Saves the trained GBDT model.
        update_threshold(roc_data, FPR): Updates the classification threshold based on the ROC curve.
    """

    def __init__(self, FPR=0.01):
        """
        Initializes a GBDTMalwareClassifier object.

        Args:
            FPR (float, optional): The desired false positive rate (FPR). Defaults to 0.01.
        """
        self.model = None
        self.threshold = None
        self._params = None
        self.fpr = FPR
    
    def predict(self, X):
        """
        Predicts the class labels for the given feature vectors.

        Args:
            X (array-like): The feature vectors.

        Returns:
            array-like: The predicted class labels.
        """
        return self.model.predict(X) > self.threshold
    
    def predict_proba(self, X):
        """
        Predicts the class probabilities for the given feature vectors.

        Args:
            X (array-like): The feature vectors.

        Returns:
            array-like: The predicted class probabilities.
        """
        return self.model.predict(X)
    
    def get_score(self, bytez):
        """
        Computes the malware score for the given raw bytes.

        Args:
            bytez (bytes): The raw bytes of the file.

        Returns:
            float: The malware score.
        """
        features = np.array(feature_extractor.feature_vector(bytez), dtype=np.float32)
        score = self.predict_proba([features])[0]
        return score

    def get_label(self, bytez):
        """
        Predicts the class label for the given raw bytes.

        Args:
            bytez (bytes): The raw bytes of the file.

        Returns:
            int: The predicted class label.
        """
        score = self.get_score(bytez)
        label = int(score > self.threshold)
        return label
    
    def print_info(self):
        """
        Prints information about the model.
        """
        print("Threshold:", self.threshold)
        print("Num trees:", self.model.num_trees())

    def train(self, X_train, y_train, X_val, y_val, FPR=0.01):
        """
        Trains the GBDT model.

        Args:
            X_train (array-like): The training feature vectors.
            y_train (array-like): The training class labels.
            X_val (array-like): The validation feature vectors.
            y_val (array-like): The validation class labels.
            FPR (float, optional): The desired false positive rate (FPR). Defaults to 0.01.
        """
        lgbm_dataset = lgb.Dataset(X_train, y_train)
        self._params = {
            "boosting": "gbdt",
            "objective": "binary",
            "num_iterations": 1000,
            "learning_rate": 0.05,
            "num_leaves": 2048,
            "max_depth": 15,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.5,
            "application": "binary"
        }
        self.model = lgb.train(self._params, lgbm_dataset)
        self.fpr = FPR

        # Set threshold based on validation set and desired FPR
        y_val_pred = self.predict_proba(X_val)
        fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)
        roc_data = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds})
        self.update_threshold(roc_data, self.fpr)

    def load(self, model_path, roc_curve_path):
        """
        Loads a pre-trained GBDT model.

        Args:
            model_path (str): The path to the model file.
            roc_curve_path (str): The path to the ROC curve data file.
        """
        self.model = lgb.Booster(model_file=model_path)
        roc_data = pd.read_csv(roc_curve_path)
        self.update_threshold(roc_data, self.fpr)

    def save(self, save_path):
        """
        Saves the trained GBDT model.

        Args:
            save_path (str): The path to save the model.
        """
        self.model.save_model(save_path)

    def update_threshold(self, roc_data, FPR):
        """
        Updates the classification threshold based on the ROC curve.

        Args:
            roc_data (pandas.DataFrame): The ROC curve data.
            FPR (float): The desired false positive rate (FPR).

        Returns:
            float: The updated classification threshold.
        """
        thr = roc_data[roc_data["fpr"] < FPR].sort_values(by="tpr", ascending=False).head(1)["thresholds"].values[0]
        self.threshold = thr
        return thr
