import lightgbm as lgb
import numpy as np
import os
from .pefeatures import PEFeatureExtractor
feature_extractor =  PEFeatureExtractor(2)

#MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
model_path = "data/EXE_scanner-GBDT-dataset=EMBER_benign_test-AE_all.txt"

class EXEscanner:
    def __init__(self, name="EXE_scanner-GBDT-dataset=EMBER_benign_test-AE_all", model_path=model_path, threshold=0.9999941463527008):
        self.name = name
        self.threshold = threshold
        self.model = lgb.Booster(model_file=model_path)

        self.print_info()

    # Works on feature vectors
    def predict(self, X):
        return self.model.predict(X) > self.threshold
    
    # Works on feature vectors
    def predict_proba(self, X):
         return self.model.predict(X)
    
    def get_score(self, bytez):
        features = np.array(feature_extractor.feature_vector(bytez), dtype=np.float32)
        score =  self.predict_proba([features])[0]
        return score

    def get_label(self, bytez):
        score = self.get_score(bytez)
        label = int(score > self.threshold)
        return label
    
    def print_info(self):
        print("Name:", self.name)
        print("Threshold:", self.threshold)
        print("Num trees:", self.model.num_trees())
