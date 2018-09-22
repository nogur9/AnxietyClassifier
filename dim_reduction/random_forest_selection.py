import os

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


def save_feature_importance(path,feature_importance):
    print(feature_importance)
    with open(os.path.join(path, "feature_importance.txt"), "w") as f:
        for i in feature_importance:
            f.write(str(i) + "\n")

def get_k_most_important_features(X,Y, k , feature_importance):
    importance_zip = sorted(zip(feature_importance,X.T), key=lambda x: x[0])
    include_columns = [importance_zip[i][1] for i in range(k)]
    highest_features = pd.DataFrame(include_columns).T
    return pd.concat([highest_features,Y],axis=1)

def get_feature_importance(X, Y):
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    return model.feature_importances_