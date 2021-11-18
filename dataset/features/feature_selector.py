from .data_features_1 import DataFeatures1
from .data_features_2 import DataFeatures2

class SelectFeatures:

    def get_feature_set(self, feature_set: int):
        
        if feature_set == 1:
            return DataFeatures1()
        elif feature_set == 2:
            return DataFeatures2()
        else:
            return None

