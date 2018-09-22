import os

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


def save_feature_importance(path,feature_importance):
    print(feature_importance)
    with open(os.path.join(path, "feature_importance.txt"), "w") as f:
        for i in feature_importance:
            f.write(str(i) + "\n")

def get_k_most_important_features(X,Y, k , feature_importance):
    feature_importance = sorted(feature_importance)
    include_columns = [feature_importance[i]["column_name"] for i in range(k)]
    highest_features = pd.DataFrame(X[include_columns])
    return pd.concat([highest_features,Y])

def get_feature_importance(X, Y):
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    return model.feature_importances_