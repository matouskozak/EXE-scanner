from malware_classifier import GBDTMalwareClassifier
import numpy as np
from pefeatures import PEFeatureExtractor

feature_extractor =  PEFeatureExtractor(2)
class MalwareDetectorWrapper:
    """
    A wrapper class for a malware detection model.
    Attributes:
        model: The machine learning model used for malware detection.
        threshold: The threshold value for classifying a file as malware.
    Methods:
        get_score(bytez):
            Computes the probability score of the given byte sequence being malware.
        get_label(bytez):
            Classifies the given byte sequence as malware or not based on the threshold.
    """
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def get_score(self, bytez):
        features = np.array(feature_extractor.feature_vector(bytez), dtype=np.float32)
        score = self.model.predict_proba([features])[0]
        return score
    
    def get_label(self, bytez):
        score = self.get_score(bytez)
        label = int(score > self.threshold)
        return label
    
EXE_SCANNER_THRESHOLDS = {
    0.01: 0.9999999950656784,
    0.001: 0.999999999388658,
    0.0001: 0.999999999388658,
}
class EXEScanner(MalwareDetectorWrapper):
    """
    EXEScanner is a specialized malware detection wrapper for scanning executable files.

    Attributes:
        target_fpr (float): The target false positive rate for the malware detection model.
        model (GBDTMalwareClassifier): The gradient boosting decision tree classifier used for malware detection.
        threshold (float): The detection threshold corresponding to the target false positive rate.

    Methods:
        __init__(target_fpr=0.01): Initializes the EXEScanner with a specified target false positive rate.
    """
    def __init__(self, target_fpr=0.01):
        model = GBDTMalwareClassifier("EXE_scanner_GBDT-dataset=benign-AE_all-before_2022", is_trained=True)
        threshold = EXE_SCANNER_THRESHOLDS[target_fpr]
        super().__init__(model, threshold)

