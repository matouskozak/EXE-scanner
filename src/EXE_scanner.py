import lightgbm as lgb
import numpy as np
import os
from .pefeatures import PEFeatureExtractor
feature_extractor =  PEFeatureExtractor(2)

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

class EXE_scanner:
    def __init__(self, name="EXE_scanner-GBDT-dataset=EMBER_benign_test-AE_all", model_path=MODULE_PATH):
        self.name = name
        self.threshold = None
        self.model = None

        self._load_model(model_path)
        self.print_info()

    def _load_model(self, model_path):
        self.model = lgb.Booster(model_file=os.path.join(model_path, f"{self.name}.txt"))
        self.threshold = 0.9999941463527008 # Pre-calculated threshold

    # Works on feature vectors
    # def predict(self, X):
    #     return self.model.predict(X) > self.threshold
    
    def predict_proba(self, X):
         return self.model.predict(X)
    
    def get_score(self, bytez):
        features = np.array(feature_extractor.feature_vector(bytez), dtype=np.float32)
        score =  self.predict_proba([features])[0]
        return score

    
    def print_info(self):
        print("Name:", self.name)
        print("Threshold:", self.threshold)
        print("Num trees:", self.model.num_trees())

    def get_label(self, bytez):
        score = self.get_score(bytez)
        label = int(score > self.threshold)
        return label

# Main classifier must expose a get_label method that returns a label (0 or 1)
# and takes a bytez argument
class Pipeline_EXE_scanner:
    def __init__(self, main_classifier, main_classifier_threshold=None, EXE_scanner=EXE_scanner()):
        self.main_classifier = main_classifier
        self.EXE_scanner = EXE_scanner
        self.main_classifier_threshold = main_classifier_threshold

    def predict_sample(self, bytez):
        if self.main_classifier.get_label(bytez) != 1.0:
            return self.EXE_scanner.get_label(bytez)        
        else:
            return 1.0
        
    def predict_sample_score(self, bytez):
        assert self.main_classifier_threshold is not None, "Main classifier's threshold must be set in the constructor"

        main_classifier_score = self.main_classifier.get_score(bytez)
        if main_classifier_score < self.main_classifier_threshold:
            return self.EXE_scanner.get_score(bytez)
        else:
            return main_classifier_score
    
        
    

